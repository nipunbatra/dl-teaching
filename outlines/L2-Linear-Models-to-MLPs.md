---
title: "Part I — Where we left off"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

## Lecture 2: From Linear Models to MLPs

**Title:** *From Linear Regression to Neural Networks*  
Core story:

\[
\boxed{
\text{linear model}
\rightarrow
\text{logistic regression}
\rightarrow
\text{neuron}
\rightarrow
\text{MLP}
\rightarrow
\text{universal approximation}
}
\]

Target: **80 min, ~48 slides**.

---

# Part I — Where we left off

### Slide 1 — Title

**From Linear Models to Neural Networks**

Subtitle:

> Linear regression, logistic regression, neurons, MLPs and universal approximation

---

### Slide 2 — Recall from Lecture 1

\[
\text{Loss} = -\log p_\theta(y\mid x)
\]

\[
\text{MSE} \leftrightarrow \text{Gaussian likelihood}
\]

\[
\text{CE} \leftrightarrow \text{Bernoulli/Categorical likelihood}
\]

Today:

\[
p_\theta(y\mid x)
\quad\text{will be parameterized by neural networks.}
\]

---

### Slide 3 — The key transition

Old ML models:

\[
f_\theta(x)=w^\top x+b
\]

Neural networks:

\[
f_\theta(x)
=
W_L\phi(W_{L-1}\phi(\cdots \phi(W_1x+b_1)))+b_L
\]

Main question:

> What changes when we replace one affine map by many affine maps plus nonlinearities?

---

# Part II — Linear regression

### Slide 4 — Linear regression model

For \(x\in\mathbb R^d\):

\[
\hat y=w^\top x+b
\]

or with bias absorbed:

\[
\hat y=\theta^\top \tilde x,
\qquad
\tilde x=
\begin{bmatrix}
1\\x
\end{bmatrix}.
\]

Visual: plane fitting data.

---

### Slide 5 — Probabilistic view

Assume:

\[
Y\mid x,w,b
\sim
\mathcal N(w^\top x+b,\sigma^2).
\]

Then:

\[
-\log p(y\mid x,w,b)
=
\frac{(y-w^\top x-b)^2}{2\sigma^2}+C.
\]

So linear regression + MSE is Gaussian MLE.

---

### Slide 6 — Matrix form

Dataset:

\[
X\in\mathbb R^{n\times d},
\qquad
y\in\mathbb R^n.
\]

Prediction:

\[
\hat y=Xw+b\mathbf 1.
\]

With bias absorbed:

\[
\hat y=\tilde X\theta.
\]

Loss:

\[
\mathcal L(\theta)
=
\frac1n\|y-\tilde X\theta\|_2^2.
\]

---

### Slide 7 — Geometry of linear regression

Visual:

- data points;
- fitted line/plane;
- residuals as vertical segments.

Message:

\[
\text{linear regression learns a hyperplane in input space.}
\]

For \(d=1\): line.  
For \(d=2\): plane.  
For \(d\): hyperplane.

---

### Slide 8 — Closed-form solution

If \(\tilde X^\top \tilde X\) is invertible:

\[
\hat\theta
=
(\tilde X^\top\tilde X)^{-1}\tilde X^\top y.
\]

But in deep learning we usually use gradient-based optimization.

Why?

- huge parameter count;
- nonlinear models;
- minibatches;
- streaming data.

---

### Slide 9 — Gradient descent preview

For MSE:

\[
\mathcal L(\theta)=\frac1n\|y-\tilde X\theta\|_2^2.
\]

Gradient:

\[
\
abla_\theta \mathcal L
=
-\frac2n\tilde X^\top(y-\tilde X\theta).
\]

Update:

\[
\theta\leftarrow\theta-\eta\
abla_\theta\mathcal L.
\]

This will become backpropagation later.

---

### Slide 10 — Interactive 1: linear regression

Demo: `linear-regression.html`

Controls:

- slope \(w\);
- intercept \(b\);
- noise level;
- learning rate;
- gradient descent steps.

Show:

- data;
- fitted line;
- MSE surface;
- parameter trajectory.

Question:

> What happens when learning rate is too large?

---

# Part III — Logistic regression

### Slide 11 — Regression output is not a probability

Linear score:

\[
z=w^\top x+b
\]

But for binary classification we need:

\[
p(y=1\mid x)\in[0,1].
\]

Solution:

\[
p(y=1\mid x)=\sigma(z).
\]

---

### Slide 12 — Sigmoid function

\[
\sigma(z)=\frac1{1+e^{-z}}.
\]

Properties:

\[
\sigma(z)\in(0,1)
\]

\[
\sigma(0)=0.5
\]

\[
\sigma(-z)=1-\sigma(z)
\]

Visual: sigmoid curve.

---

### Slide 13 — Logistic regression model

\[
z=w^\top x+b
\]

\[
p_\theta(y=1\mid x)=\sigma(z)
\]

\[
p_\theta(y=0\mid x)=1-\sigma(z)
\]

So:

\[
Y\mid x
\sim
\operatorname{Bernoulli}(\sigma(w^\top x+b)).
\]

---

### Slide 14 — Binary cross-entropy loss

For \(y\in\{0,1\}\):

\[
\mathcal L
=
-y\log \hat p
-
(1-y)\log(1-\hat p)
\]

where

\[
\hat p=\sigma(w^\top x+b).
\]

Equivalent compact form:

\[
p(y\mid x)
=
\hat p^y(1-\hat p)^{1-y}.
\]

---

### Slide 15 — Decision boundary

Classifier:

\[
\hat y=
\begin{cases}
1, & \sigma(w^\top x+b)\ge 0.5,\\
0, & \text{otherwise}.
\end{cases}
\]

Since

\[
\sigma(z)\ge 0.5
\iff
z\ge 0,
\]

decision boundary is:

\[
w^\top x+b=0.
\]

Visual: 2D points separated by a line.

---

### Slide 16 — Logistic regression is still linear

Even though sigmoid is nonlinear, the decision boundary is linear:

\[
w^\top x+b=0.
\]

Important distinction:

- output probability is nonlinear in \(z\);
- decision boundary is linear in \(x\).

Message:

\[
\boxed{\text{logistic regression = linear classifier + sigmoid}}
\]

---

### Slide 17 — Gradient intuition

For one example:

\[
\hat p=\sigma(w^\top x+b)
\]

\[
\mathcal L=
-y\log\hat p-(1-y)\log(1-\hat p)
\]

The beautiful result:

\[
\frac{\partial \mathcal L}{\partial z}
=
\hat p-y.
\]

Then:

\[
\
abla_w\mathcal L=(\hat p-y)x.
\]

---

### Slide 18 — Interactive 2: logistic regression

Demo: `logistic-regression.html`

Controls:

- line angle;
- bias;
- sigmoid steepness;
- threshold;
- add/remove points.

Show:

- probability heatmap;
- decision boundary;
- BCE loss;
- misclassified points.

Question:

> How does the decision boundary move when \(b\) changes?

---

# Part IV — From logistic regression to a neuron

### Slide 19 — A neuron

A neuron computes:

\[
z=w^\top x+b
\]

\[
a=\phi(z).
\]

Diagram:

```text
x1 --w1--
x2 --w2--  →  Σ + b  →  activation  →  a
xd --wd--
```

This is the basic unit of a neural network.

---

### Slide 20 — Logistic regression is one neuron

For binary classification:

\[
a=\sigma(w^\top x+b)
\]

\[
a=p(y=1\mid x).
\]

So logistic regression is a single sigmoid neuron.

Message:

\[
\boxed{\text{logistic regression is the simplest neural network}}
\]

---

### Slide 21 — Common activation functions

Show curves:

\[
\sigma(z)=\frac1{1+e^{-z}}
\]

\[
\tanh(z)
\]

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

\[
\operatorname{GELU}(z)\approx z\Phi(z)
\]

Mention:

- sigmoid: probabilities/gates;
- tanh: zero-centered;
- ReLU: common hidden activation;
- GELU: common in Transformers.

---

### Slide 22 — Why activation is needed

If we stack only linear maps:

\[
h_1=W_1x+b_1
\]

\[
h_2=W_2h_1+b_2
\]

then:

\[
h_2=W_2W_1x+(W_2b_1+b_2).
\]

This is still linear.

Therefore:

\[
\boxed{\text{without nonlinear activation, depth collapses to one linear layer}}
\]

---

### Slide 23 — Interactive 3: activation functions

Demo: `activation-functions.html`

Controls:

- activation type;
- input \(z\);
- show derivative;
- compare saturation.

Show:

- \(\phi(z)\);
- \(\phi'(z)\);
- effect on gradient.

Questions:

1. Where does sigmoid saturate?
2. Where is ReLU derivative zero?
3. Why might saturation hurt training?

---

# Part V — Multi-class classification

### Slide 24 — Binary to multi-class

Binary:

\[
z=w^\top x+b
\]

\[
p(y=1\mid x)=\sigma(z).
\]

Multi-class:

\[
z_k=w_k^\top x+b_k,
\qquad k=1,\dots,K.
\]

Each class gets a score/logit.

---

### Slide 25 — Softmax regression

For \(K\) classes:

\[
p(y=k\mid x)
=
\frac{\exp(z_k)}
{\sum_{j=1}^K\exp(z_j)}.
\]

where

\[
z=Wx+b.
\]

This is also called multinomial logistic regression.

---

### Slide 26 — Multi-class cross-entropy

For one-hot target \(y\):

\[
\mathcal L
=
-\sum_{k=1}^K y_k\log p_k.
\]

Since one \(y_k=1\):

\[
\mathcal L=-\log p_{\text{true class}}.
\]

---

### Slide 27 — Geometry of softmax regression

For classes \(i\) and \(j\), boundary occurs when:

\[
z_i=z_j.
\]

That is:

\[
w_i^\top x+b_i
=
w_j^\top x+b_j.
\]

So:

\[
(w_i-w_j)^\top x+(b_i-b_j)=0.
\]

Decision regions are linear/polyhedral.

Visual: 3-class linear regions in 2D.

---

### Slide 28 — Softmax regression limitation

Softmax regression can only produce linear decision boundaries.

It fails on:

- XOR;
- concentric circles;
- spiral data;
- complex image/audio/text patterns.

Visual: XOR dataset with no single line separation.

---

### Slide 29 — Interactive 4: softmax classifier

Demo: `softmax-regression.html`

Controls:

- three class weight vectors;
- bias terms;
- input point;
- temperature \(T\).

Show:

- logits;
- softmax probabilities;
- decision regions;
- CE loss.

Question:

> Why are the decision regions straight-edged?

---

# Part VI — Multilayer perceptron

### Slide 30 — Add a hidden layer

Instead of:

\[
z=Wx+b
\]

use:

\[
h=\phi(W_1x+b_1)
\]

\[
z=W_2h+b_2.
\]

For classification:

\[
p=\operatorname{softmax}(z).
\]

This is a one-hidden-layer MLP.

---

### Slide 31 — MLP architecture diagram

Draw:

```text
input layer → hidden layer → output layer
```

With dimensions:

\[
x\in\mathbb R^d
\]

\[
W_1\in\mathbb R^{m\times d}
\]

\[
h\in\mathbb R^m
\]

\[
W_2\in\mathbb R^{K\times m}
\]

\[
z\in\mathbb R^K
\]

---

### Slide 32 — Each hidden neuron creates a feature

Hidden unit:

\[
h_j=\phi(w_j^\top x+b_j).
\]

Interpretation:

- first layer learns features;
- second layer combines features.

For ReLU:

\[
h_j=\max(0,w_j^\top x+b_j).
\]

Each ReLU is active on one side of a hyperplane.

---

### Slide 33 — Feature transformation view

Original input space:

\[
x
\]

Hidden representation:

\[
h=\phi(W_1x+b_1)
\]

Classifier works in hidden space:

\[
z=W_2h+b_2.
\]

Message:

\[
\boxed{
\text{MLP learns a representation where classification becomes easier}
}
\]

Visual:

- XOR in input space;
- transformed hidden space where it is linearly separable.

---

### Slide 34 — MLP for binary classification

Architecture:

\[
h=\phi(W_1x+b_1)
\]

\[
z=w_2^\top h+b_2
\]

\[
p(y=1\mid x)=\sigma(z).
\]

Loss:

\[
\mathcal L
=
-y\log p-(1-y)\log(1-p).
\]

---

### Slide 35 — MLP for multi-class classification

Architecture:

\[
h=\phi(W_1x+b_1)
\]

\[
z=W_2h+b_2
\]

\[
p=\operatorname{softmax}(z).
\]

Loss:

\[
\mathcal L=-\log p_y.
\]

Same output layer as softmax regression; only representation changed.

---

### Slide 36 — Multiple hidden layers

General MLP:

\[
h^{(1)}=\phi(W_1x+b_1)
\]

\[
h^{(2)}=\phi(W_2h^{(1)}+b_2)
\]

\[
\cdots
\]

\[
z=W_Lh^{(L-1)}+b_L.
\]

Then:

\[
p=\operatorname{softmax}(z)
\]

or

\[
\hat y=z
\]

for regression.

---

### Slide 37 — Depth vs width

Width:

\[
\text{number of neurons per layer}
\]

Depth:

\[
\text{number of layers}
\]

Intuition:

- width gives more features at same level;
- depth composes features;
- deep models build hierarchical representations.

Examples:

- pixels → edges → motifs → objects;
- characters → words → phrases → semantics.

---

### Slide 38 — Interactive 5: MLP decision boundaries

Demo: `mlp-decision-boundary.html`

Datasets:

- linearly separable;
- XOR;
- circles;
- spirals.

Controls:

- number of hidden units;
- number of layers;
- activation;
- training steps;
- learning rate.

Show:

- decision boundary;
- train loss;
- accuracy;
- hidden unit activations.

Question:

> What changes when we add hidden units? What changes when we add layers?

---

# Part VII — Why MLPs can represent complex functions

### Slide 39 — Piecewise linear view of ReLU networks

For ReLU:

\[
\phi(z)=\max(0,z).
\]

A ReLU network is piecewise linear.

Visual:

- one ReLU: hinge;
- sum of ReLUs: piecewise linear curve;
- many ReLUs: complex approximation.

---

### Slide 40 — Building functions from ReLUs

One-hidden-layer scalar network:

\[
f(x)=
\sum_{j=1}^{m}
a_j\operatorname{ReLU}(w_jx+b_j)+c.
\]

Each hidden unit adds one kink.

With enough hidden units, we can approximate complicated 1D functions.

Visual: approximate sine curve using sum of ReLUs.

---

### Slide 41 — Universal approximation theorem

Informal statement:

Let \(f\) be a continuous function on a compact domain.  
A feedforward neural network with one hidden layer and suitable nonlinear activation can approximate \(f\) arbitrarily well.

Symbolically:

\[
\forall \epsilon>0,\quad
\exists\ \text{MLP } g
\quad\text{s.t.}\quad
\sup_x |f(x)-g(x)|<\epsilon.
\]

---

### Slide 42 — What UAT does not say

Universal approximation does **not** say:

- training will find the function;
- finite data is enough;
- small network is enough;
- generalization is guaranteed;
- one hidden layer is computationally efficient.

Message:

\[
\boxed{
\text{expressivity} \
eq \text{trainability} \
eq \text{generalization}
}
\]

---

### Slide 43 — Why depth still matters

Even if one hidden layer is universal, depth can be exponentially more efficient for some functions.

Intuition:

\[
\text{composition is natural}
\]

\[
f(x)=f_L\circ f_{L-1}\circ\cdots\circ f_1(x).
\]

Examples:

- image hierarchy;
- language hierarchy;
- program-like computations.

---

### Slide 44 — Interactive 6: function approximation

Demo: `function-approximation.html`

Controls:

- target function: sine, square wave, polynomial, custom;
- hidden units;
- layers;
- activation;
- training iterations.

Show:

- target function;
- network approximation;
- residual;
- number of parameters.

Question:

> Does more expressivity always mean easier optimization?

---

# Part VIII — Connecting to deep learning practice

### Slide 45 — One training example through an MLP

Forward pass:

\[
x
\rightarrow
h^{(1)}
\rightarrow
h^{(2)}
\rightarrow
z
\rightarrow
p
\rightarrow
\mathcal L.
\]

For multi-class:

\[
\mathcal L=-\log p_y.
\]

Diagram: computation graph.

This prepares for backpropagation.

---

### Slide 46 — Parameters of an MLP

For layer \(\ell\):

\[
W_\ell\in\mathbb R^{d_\ell\times d_{\ell-1}},
\qquad
b_\ell\in\mathbb R^{d_\ell}.
\]

Total parameters:

\[
\sum_{\ell=1}^{L}
(d_\ell d_{\ell-1}+d_\ell).
\]

Quick exercise:

Input \(784\), hidden \(128\), output \(10\):

\[
784\times128+128+128\times10+10.
\]

---

### Slide 47 — The standard supervised DL template

Regression:

\[
z=f_\theta(x),
\qquad
\mathcal L=(y-z)^2.
\]

Binary classification:

\[
z=f_\theta(x),
\qquad
p=\sigma(z),
\qquad
\mathcal L=\text{BCE}.
\]

Multi-class classification:

\[
z=f_\theta(x),
\qquad
p=\operatorname{softmax}(z),
\qquad
\mathcal L=\text{CE}.
\]

---

### Slide 48 — Summary table

| Model | Output | Distribution | Loss | Boundary |
|---|---|---|---|---|
| Linear regression | \(w^\top x+b\) | Gaussian | MSE | — |
| Logistic regression | \(\sigma(w^\top x+b)\) | Bernoulli | BCE | linear |
| Softmax regression | \(\operatorname{softmax}(Wx+b)\) | Categorical | CE | linear |
| MLP classifier | \(\operatorname{softmax}(f_\theta(x))\) | Categorical | CE | nonlinear |

---

### Slide 49 — Final mental model

\[
\boxed{
\text{linear regression} =
\text{linear score + Gaussian likelihood}
}
\]

\[
\boxed{
\text{logistic regression} =
\text{linear score + sigmoid + Bernoulli likelihood}
}
\]

\[
\boxed{
\text{softmax regression} =
\text{linear scores + softmax + categorical likelihood}
}
\]

\[
\boxed{
\text{MLP} =
\text{learned nonlinear features + output likelihood}
}
\]

---

### Slide 50 — Next lecture

Now we have:

\[
x\rightarrow f_\theta(x)\rightarrow \mathcal L.
\]

Next:

\[
\frac{\partial \mathcal L}{\partial \theta}
\]

Topics:

- computation graphs;
- chain rule;
- backpropagation;
- gradient descent;
- automatic differentiation.

---

## Timing

| Section | Slides | Time |
|---|---:|---:|
| Recap and motivation | 1–3 | 5 min |
| Linear regression | 4–10 | 12 min |
| Logistic regression | 11–18 | 14 min |
| Neuron and activations | 19–23 | 10 min |
| Softmax regression | 24–29 | 10 min |
| MLPs | 30–38 | 16 min |
| Universal approximation | 39–44 | 10 min |
| DL template and summary | 45–50 | 8 min |

Total: **~85 min nominal**.  
Skim slides **8, 17, 43, 46** if short on time.

---

## Interactives to make

1. `linear-regression.html`  
   MSE surface, gradient descent path, fitted line.

2. `logistic-regression.html`  
   Sigmoid probability heatmap and linear decision boundary.

3. `activation-functions.html`  
   Sigmoid, tanh, ReLU, GELU and derivatives.

4. `softmax-regression.html`  
   Multi-class logits, softmax probabilities, linear decision regions.

5. `mlp-decision-boundary.html`  
   XOR/circles/spiral with hidden units/layers slider.

6. `function-approximation.html`  
   ReLU MLP approximating 1D target functions.

Framework: **Typst + Touying for slides; standalone JS demos on GitHub Pages.**
