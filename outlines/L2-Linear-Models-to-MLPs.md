---
title: "Lecture 2 — From Linear Models to MLPs"
status: "Synchronized with the current public deck"
duration: "80 minutes"
---

# Lecture 2: From Linear Models to MLPs

## Core story

$$
\boxed{
\text{affine prediction}
\rightarrow
\text{nonlinear feature}
\rightarrow
\text{XOR counterexample}
\rightarrow
\text{ReLU MLP}
\rightarrow
\text{universal approximation}
\rightarrow
\text{depth as composition}
}
$$

Lecture 1 chose a probabilistic output and its loss. Lecture 2 keeps that loss
and enriches the prediction rule $f_\theta(x)$. The final handoff is deliberately
short:

$$
\text{representation}\rightarrow\text{loss}\rightarrow
\text{backpropagation}\rightarrow\text{optimizer}.
$$

This outline follows the current deck by teaching beat rather than by PDF page.
Reveal states can change the presentation page count, so page-number references
are intentionally omitted.

## Learning outcomes

By the end, students should be able to:

1. Compute one weighted sum and its batch-matrix form.
2. Distinguish pre-activation $u$ from activation $h=\phi(u)$.
3. Explain why a stack of affine maps is still affine.
4. Explain why one straight boundary cannot separate XOR.
5. Evaluate the fixed two-ReLU XOR network on all four Boolean inputs.
6. Distinguish four-point XOR from filled-region XOR.
7. Read ReLU units as shifted hinges whose weighted sum builds a
   piecewise-linear function.
8. State universal approximation as an existence result on a specified region.
9. Explain width as parallel variety and depth as sequential composition.
10. Count a two-layer MLP's parameters and connect representation learning to
    loss, backpropagation, and optimization.

# Part I — Change the prediction rule

## Lecture 1 chose the loss; Lecture 2 expands the model

Keep from Lecture 1:

- Gaussian outputs lead to MSE for regression.
- Bernoulli or categorical outputs lead to cross-entropy for classification.

Change today:

$$
\hat y=f_\theta(x)
$$

moves from one affine map to affine maps separated by nonlinear activations.

## One notation throughout

- $x$: input features;
- $\theta$: every trainable weight and bias;
- $f_\theta$: the prediction rule;
- output: a regression value, logit, or vector of class scores.

Do not begin with the fully nested neural-network formula. Build it one operation
at a time.

# Part II — Linear prediction

## One weighted sum

Start with

$$
\hat y=w^Tx+b.
$$

Use the node diagram to make every edge contribution explicit:

$$
w_jx_j,\qquad b\cdot1,\qquad
\hat y=\sum_j w_jx_j+b.
$$

## A dataset is a labelled matrix

For $n$ examples and $d$ features,

$$
X\in\mathbb R^{n\times d},\qquad y\in\mathbb R^n,
$$

and the same parameters process every row:

$$
\hat y=Xw+b\mathbf{1}.
$$

Absorbing the bias gives

$$
\tilde X=[X\ \mathbf{1}],\qquad
\theta=\begin{bmatrix}w\\b\end{bmatrix},\qquad
\hat y=\tilde X\theta.
$$

The point of this sequence is shape literacy and parameter reuse.

## One affine model gives one flat trend

In one input dimension the graph is a line; with two input dimensions it is a
plane. The rule can tilt and shift, but it cannot bend.

The corresponding slide links the
[Linear GD Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L02/01_linear_regression_gd.ipynb).
The notebook uses seeded synthetic data and compares gradient descent with the
closed-form solution.

# Part III — From a score to a neuron

## Two operations, not one

$$
u=w^Tx+b,
\qquad
h=\phi(u).
$$

Keep $u$ and $h$ visually separate. The bias is part of the affine operation;
the activation creates the nonlinear feature.

## Activation families

- Sigmoid and tanh act like bounded, smooth switches.
- ReLU and GELU act like ramps.
- Saturation means that a curve is locally flat, so a change in input produces
  only a small change in output.

Use the worked saturation calculation before opening the
[activation explorer](https://nipunbatra.github.io/dl-teaching/interactives/activation-explorer.html).

## Why affine depth collapses

Ask first whether two affine layers can make a bend. Then reveal

$$
W_2(W_1x+b_1)+b_2
=(W_2W_1)x+(W_2b_1+b_2).
$$

Without an intervening activation, the middle layer adds parameters but not a
new function shape.

# Part IV — Where one weighted sum fails

## Binary classification

A binary affine classifier changes class where

$$
w^Tx+b=0,
$$

which is a straight line in two dimensions. XOR places equal labels at opposite
corners, so no single line separates it.

## Multiclass classification

Each class has an affine score

$$
z_k(x)=w_k^Tx+b_k.
$$

Softmax changes scores into probabilities but preserves their order. A candidate
tie between classes $i$ and $j$ satisfies

$$
z_i(x)=z_j(x),
$$

which is a flat set. Only the portion where both tied scores are maximal is a
visible decision boundary. Use the explicit points to distinguish candidate
ties from visible boundaries; do not present them as a benchmark dataset.

# Part V — The multilayer perceptron and XOR

## A hidden layer is a learned feature map

$$
h=\phi(W_1x+b_1),
\qquad
z=W_2h+b_2.
$$

For input dimension $d$, hidden width $m$, and $K$ outputs:

$$
W_1\in\mathbb R^{m\times d},
\quad
b_1\in\mathbb R^m,
\quad
W_2\in\mathbb R^{K\times m},
\quad
b_2\in\mathbb R^K.
$$

Move the parameter count here, while the architecture is visible:

$$
\underbrace{md+m}_{\text{hidden layer}}
+
\underbrace{Km+K}_{\text{output layer}}
=m(d+1)+K(m+1).
$$

## One exact two-ReLU construction for four-point XOR

Use the same functions throughout:

$$
h_1=\operatorname{ReLU}(x_1+x_2),
\qquad
h_2=\operatorname{ReLU}(x_1+x_2-1),
$$

$$
z=2h_1-4h_2-1,
\qquad
\hat y=\mathbf{1}[z\ge0].
$$

Required checks:

1. Evaluate all four Boolean inputs.
2. Derive $h_1(x)$ and $h_2(x)$ symbolically before substituting one point.
3. Verify the hidden calculation in matrix form.
4. Combine the features into $z$, then compute sigmoid probability and binary
   cross-entropy for one labelled example.
5. Plot the two hidden features in input space.
6. Map all four inputs into $(h_1,h_2)$ space and show why one output boundary
   can now separate them.

These are hand-chosen weights. The slides demonstrate representational capacity,
not optimizer success.

## Four points and four filled regions are different tasks

The same two-ReLU logit is positive only when

$$
0.5\le x_1+x_2\le1.5.
$$

It classifies the four Boolean corners exactly but paints a diagonal band over a
filled square, giving 75% area agreement for the filled-XOR lab setup. A separate
four-ReLU construction represents the quadrant pattern away from the two
boundary axes.

The comparison slide links the
[XOR Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L02/02_mlp_xor.ipynb).
The notebook verifies both fixed constructions on dense grids.

Open the
[ReLU Classification Lab](https://nipunbatra.github.io/interactive-articles/mlp-decision-boundary/)
to compare:

- four XOR corners and filled XOR;
- the two-ReLU corner rule and four-ReLU regional rule;
- fixed constructions and optimizer runs;
- logit, probability, class map, signed unit contributions, and the rotatable
  3-D logit surface.

Reset between a constructed rule and a training run so the evidence labels do
not blur together.

## Output structure

- Binary classification can use one output logit.
- $K$-class classification uses $K$ output scores.
- Additional hidden layers add stages of learned features.

# Part VI — Universal approximation

## The question

Given a continuous target $f$ on a specified compact domain $D$, can a finite
network $g$ make the largest error smaller than any requested tolerance?

## A ReLU is a hinge

$$
h_j(x)=\operatorname{ReLU}(w_jx+b_j).
$$

In one dimension, the zero of $w_jx+b_j$ sets the hinge location. An output
weight chooses the sign and size of the slope change. Several hidden units
therefore provide a collection of symbolic hinge features for the same input.

## Three hinges build a tent

$$
g(x)=\operatorname{ReLU}(x)
-2\operatorname{ReLU}(x-1)
+\operatorname{ReLU}(x-2).
$$

Read it interval by interval:

- flat before $0$;
- rise on $[0,1)$;
- fall on $[1,2)$;
- flat after $2$.

The output weights choose the slope changes. This concrete construction is the
bridge to the theorem.

## Beyond one-dimensional functions

One ReLU applied to a two-dimensional input folds the input plane along
$w^Tx+b=0$: one side is flat and the other rises. Multiple such features feed
class scores, producing piecewise-linear decision boundaries. Keep this to one
function slide and one classification slide; it is a conceptual extension, not
a new derivation sequence.

## Construction and training are different questions

Use the
[ReLU Approximation Lab](https://nipunbatra.github.io/interactive-articles/relu-function-approximation/)
in two lanes:

- **Construction:** choose a target, place hinges, and inspect the resulting
  function, residual, worst grid gap, and signed terms.
- **Training:** keep target and width fixed, reset the seed, and inspect what the
  optimizer finds.

More units increase representational freedom; they do not guarantee a better
optimizer outcome on every run.

## Read the error statement in order

1. Fix a continuous target $f$ and a compact region $D$.
2. At one input, define the vertical gap
   $|f(x_0)-g(x_0)|$.
3. Over the region, use
   $\sup_{x\in D}|f(x)-g(x)|$ for the worst gap.
4. Choose any $\epsilon>0$.
5. The theorem says that some finite MLP $g$ can make
   $\sup_{x\in D}|f(x)-g(x)|<\epsilon$.

The displayed interpolants use fixed knots and are **constructed, not trained**.

## What universal approximation does not say

It does not guarantee:

- that training finds the approximating network;
- that a small width suffices;
- that finite data identifies the desired function;
- that the result generalizes beyond the observed data or chosen region.

$$
\text{expressivity}\ne\text{trainability}\ne\text{generalization}.
$$

# Part VII — Depth versus width

## Node view

- Width adds more features in parallel at one stage.
- Depth adds more transformations in sequence.

## Two slides for one concrete composition

Use

$$
h_1(x)=\operatorname{ReLU}(2x+1),
\qquad
h_2(h_1)=\operatorname{ReLU}(2-h_1).
$$

First show the two maps separately and stress that the horizontal axis of the
second plot is $h_1$, not raw $x$. Then compose them:

$$
h_2(x)=\operatorname{ReLU}\!\left(2-\operatorname{ReLU}(2x+1)\right).
$$

The result has three input regions:

- $x\le-0.5$: left plateau, $h_2=2$;
- $-0.5<x<0.5$: middle ramp, $h_2=1-2x$;
- $x\ge0.5$: right plateau, $h_2=0$.

This example explains sequential composition. It does not claim that the simple
function requires a deep network.

## One hierarchy example only: vision

The retained schematic reads

$$
\text{pixels}\rightarrow\text{strokes}\rightarrow
\text{loops}\rightarrow\text{evidence for a digit}.
$$

It is a possible compositional story, not a measured activation trace.

## Decision rule

- Add width when more same-stage feature variety is useful.
- Add depth when the target is naturally described by sequential composition.

This is a modelling intuition, not a universal architecture-selection theorem.

# Part VIII — From representation to learning

## One closing handoff

$$
x
\rightarrow h^{(1)}
\rightarrow h^{(2)}
\rightarrow z
\rightarrow \mathcal L.
$$

- **Forward:** evaluate the current representation, prediction, and scalar loss.
- **Next — backpropagation:** compute how each parameter affected the loss.
- **Then — optimization:** use those gradients to change the parameters.

Hidden layers enrich $f_\theta(x)$; the output loss can remain the same one
chosen in Lecture 1. Keep this handoff to one slide.

# 80-minute route

| Time | Teaching beat | Must preserve |
|---:|---|---|
| 0–5 min | Lecture 1 handoff and notation | Change the model, keep the loss |
| 5–15 min | Weighted sums and matrix form | Shapes; bias as a ones column; Linear GD Colab link |
| 15–25 min | Neuron and activations | $u$ versus $h$; saturation; affine collapse |
| 25–35 min | Linear classification limit | XOR; softmax preserves the winner; visible ties |
| 35–52 min | MLP and XOR | Parameter count; all four rows; filled-region scope; XOR Colab and classifier lab |
| 52–67 min | Hinges and UAT | Tent construction; 2-D fold; `sup`; existence caveat; function lab |
| 67–76 min | Width and depth | Two composition slides; one vision hierarchy |
| 76–80 min | Learning handoff | Representation $\to$ loss $\to$ backprop $\to$ optimizer |

If shortened to 70 minutes, compress the explicit multiclass-score example and
the interpolant comparison. Preserve the four-row XOR verification, the
four-points-versus-filled-regions distinction, and the UAT caveat.

# Synchronized resources

| Resource | Deck location | Purpose |
|---|---|---|
| [Linear GD Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L02/01_linear_regression_gd.ipynb) | “For regression, one weighted sum gives one flat trend” | Fit seeded synthetic data and compare with the closed form |
| [XOR Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L02/02_mlp_xor.ipynb) | “Four-point XOR and filled XOR are different tasks” | Verify the two fixed XOR constructions |
| [Activation explorer](https://nipunbatra.github.io/dl-teaching/interactives/activation-explorer.html) | Activation interactive slide | Compare shape, range, and saturation |
| [ReLU Classification Lab](https://nipunbatra.github.io/interactive-articles/mlp-decision-boundary/) | Four-points-versus-regions interactive slide | Compare datasets, representations, training, and the 3-D logit surface |
| [ReLU Approximation Lab](https://nipunbatra.github.io/interactive-articles/relu-function-approximation/) | Function-approximation interactive slide | Separate constructive approximation from optimizer behaviour |

# Evidence guardrails

- Call the displayed XOR weights **constructed**, not learned.
- Call the regression data **seeded synthetic**, not real-world data.
- Call the fixed-knot approximation curves **constructed interpolants**, not
  trained models.
- Call the vision hierarchy a **conceptual schematic**, not an activation map.
- Treat UAT as a representation theorem, not an optimization or generalization
  guarantee.

# Exit ticket

For $(x_1,x_2)=(1,1)$, compute $h_1$, $h_2$, $z$, and the predicted class for the
fixed XOR MLP. Then answer:

1. Why does 4/4 on the Boolean truth table not imply perfect filled-XOR
   classification?
2. What does universal approximation promise?
3. What does it not promise about training?
