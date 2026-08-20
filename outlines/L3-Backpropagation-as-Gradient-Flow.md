---
title: "Lecture 3: Backpropagation as Gradient Flow"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 3: Backpropagation as Gradient Flow

Core vocabulary:

\[
\boxed{
\text{upstream gradient}
\times
\text{local gradient}
=
\text{downstream gradient}
}
\]

and at branches:

\[
\boxed{
\text{gradients accumulate by addition}
}
\]

Target: **80 min, ~52 slides**.

---

## Main deck outline

### Slide 1 — Title

**Computation Graphs, Backpropagation and Autodiff**

Subtitle:

> How gradients flow through neural networks

---

### Slide 2 — Goal

Forward pass:

\[
x\rightarrow f_\theta(x)\rightarrow \mathcal L
\]

Backward pass:

\[
\mathcal L\rightarrow \
abla_\theta \mathcal L
\]

Question:

\[
\text{How do we compute all parameter gradients efficiently?}
\]

---

### Slide 3 — Training needs gradients

\[
\theta_{t+1}
=
\theta_t-\eta\
abla_\theta\mathcal L(\theta_t)
\]

Today:

\[
\boxed{\text{compute }\
abla_\theta\mathcal L}
\]

Next lecture:

\[
\boxed{\text{use }\
abla_\theta\mathcal L\text{ well}}
\]

---

# Part I — Backprop vocabulary

### Slide 4 — One local computation

Consider one node:

\[
v=f(u)
\]

and eventually:

\[
v\rightarrow \mathcal L
\]

Diagram:

```text
u → [ f ] → v → ... → L
```

We want:

\[
\frac{\partial \mathcal L}{\partial u}
\]

---

### Slide 5 — Upstream, local, downstream

For:

\[
u\rightarrow v\rightarrow \mathcal L
\]

define:

\[
\underbrace{\frac{\partial \mathcal L}{\partial v}}_{\text{upstream gradient}}
\]

\[
\underbrace{\frac{\partial v}{\partial u}}_{\text{local gradient}}
\]

\[
\underbrace{\frac{\partial \mathcal L}{\partial u}}_{\text{downstream gradient}}
\]

Chain rule:

\[
\boxed{
\frac{\partial \mathcal L}{\partial u}
=
\frac{\partial \mathcal L}{\partial v}
\frac{\partial v}{\partial u}
}
\]

---

### Slide 6 — Backprop rule

\[
\boxed{
\text{downstream}
=
\text{upstream}
\times
\text{local}
}
\]

In symbols:

\[
\bar u
=
\bar v
\frac{\partial v}{\partial u}
\]

where:

\[
\bar u
=
\frac{\partial \mathcal L}{\partial u},
\qquad
\bar v
=
\frac{\partial \mathcal L}{\partial v}.
\]

This bar notation is common in autodiff.

---

### Slide 7 — Example: square node

Forward:

\[
v=u^2
\]

Local gradient:

\[
\frac{\partial v}{\partial u}=2u
\]

If upstream gradient is:

\[
\bar v=7
\]

then downstream gradient is:

\[
\bar u
=
7\cdot 2u.
\]

At \(u=3\):

\[
\bar u=42.
\]

---

### Slide 8 — Example: addition node

Forward:

\[
v=a+b
\]

Local gradients:

\[
\frac{\partial v}{\partial a}=1,
\qquad
\frac{\partial v}{\partial b}=1.
\]

Given upstream \(\bar v\):

\[
\bar a=\bar v,
\qquad
\bar b=\bar v.
\]

Addition copies gradient backward.

---

### Slide 9 — Example: multiplication node

Forward:

\[
v=ab
\]

Local gradients:

\[
\frac{\partial v}{\partial a}=b,
\qquad
\frac{\partial v}{\partial b}=a.
\]

Given upstream \(\bar v\):

\[
\bar a=\bar v b,
\qquad
\bar b=\bar v a.
\]

Multiplication sends “the other input” backward.

---

### Slide 10 — Activation node

Forward:

\[
v=\phi(u)
\]

Backward:

\[
\bar u=\bar v\phi'(u).
\]

Examples:

\[
\sigma'(u)=\sigma(u)(1-\sigma(u))
\]

\[
\operatorname{ReLU}'(u)=
\begin{cases}
1,&u>0\\
0,&u<0
\end{cases}
\]

---

### Slide 11 — Rule table

| Forward op | Local gradient | Backward update |
|---|---|---|
| \(v=a+b\) | \(1,1\) | \(\bar a{+}{=}\bar v,\ \bar b{+}{=}\bar v\) |
| \(v=ab\) | \(b,a\) | \(\bar a{+}{=}\bar v b,\ \bar b{+}{=}\bar v a\) |
| \(v=u^2\) | \(2u\) | \(\bar u{+}{=}\bar v 2u\) |
| \(v=\phi(u)\) | \(\phi'(u)\) | \(\bar u{+}{=}\bar v\phi'(u)\) |

Important: use `+=`, not `=`.

This prepares gradient accumulation.

---

# Part II — Gradient accumulation

### Slide 12 — One variable, multiple paths

Suppose:

\[
x\rightarrow u\rightarrow \mathcal L
\]

and

\[
x\rightarrow v\rightarrow \mathcal L.
\]

Then:

\[
\frac{\partial \mathcal L}{\partial x}
=
\frac{\partial \mathcal L}{\partial u}
\frac{\partial u}{\partial x}
+
\frac{\partial \mathcal L}{\partial v}
\frac{\partial v}{\partial x}.
\]

---

### Slide 13 — Gradients add at branches

\[
\boxed{
\text{If a variable influences the loss through multiple paths,}
}
\]

\[
\boxed{
\text{the total gradient is the sum of path contributions.}
}
\]

Diagram:

```text
      → u →
x ───       → L
      → v →
```

Backward:

```text
      ← contribution 1
x ← sum
      ← contribution 2
```

---

### Slide 14 — Worked branch example

\[
u=x^2
\]

\[
v=3x
\]

\[
L=u+v
\]

So:

\[
L=x^2+3x.
\]

Analytically:

\[
\frac{dL}{dx}=2x+3.
\]

At \(x=4\):

\[
\frac{dL}{dx}=11.
\]

---

### Slide 15 — Same example by backprop

Forward at \(x=4\):

\[
u=x^2=16
\]

\[
v=3x=12
\]

\[
L=u+v=28.
\]

Backward:

\[
\bar L=1
\]

Since \(L=u+v\):

\[
\bar u=1,
\qquad
\bar v=1.
\]

From \(u=x^2\):

\[
\bar x_u=\bar u\cdot 2x=1\cdot8=8.
\]

From \(v=3x\):

\[
\bar x_v=\bar v\cdot 3=3.
\]

Accumulate:

\[
\bar x=\bar x_u+\bar x_v=8+3=11.
\]

---

### Slide 16 — Why `+=` matters

In autodiff, each variable stores an adjoint:

\[
\bar x=\frac{\partial L}{\partial x}.
\]

When a new backward path contributes:

\[
\bar x\leftarrow \bar x+\text{new contribution}.
\]

Not:

\[
\bar x\leftarrow \text{new contribution}.
\]

This is why gradients accumulate in PyTorch unless cleared.

---

# Part III — Fully worked example

Use this as the central 15-minute derivation.

### Slide 17 — Worked example: scalar linear model

Model:

\[
\hat y=wx+b
\]

Loss:

\[
L=(\hat y-y)^2
\]

Choose values:

\[
x=3,\quad w=2,\quad b=1,\quad y=10.
\]

---

### Slide 18 — Break into primitive operations

\[
m=wx
\]

\[
a=m+b
\]

\[
e=a-y
\]

\[
L=e^2
\]

Graph:

```text
w ─┐
   × → m ─┐
x ─┘      + → a ─┐
          b ────┘
                 - → e → square → L
          y ─────┘
```

---

### Slide 19 — Forward pass table

| Node | Formula | Value |
|---|---:|---:|
| \(x\) | given | \(3\) |
| \(w\) | given | \(2\) |
| \(b\) | given | \(1\) |
| \(y\) | given | \(10\) |
| \(m\) | \(wx\) | \(6\) |
| \(a\) | \(m+b\) | \(7\) |
| \(e\) | \(a-y\) | \(-3\) |
| \(L\) | \(e^2\) | \(9\) |

---

### Slide 20 — Start backward

Always start with:

\[
\bar L
=
\frac{\partial L}{\partial L}
=
1.
\]

Node:

\[
L=e^2.
\]

Local gradient:

\[
\frac{\partial L}{\partial e}=2e.
\]

At \(e=-3\):

\[
\frac{\partial L}{\partial e}=-6.
\]

Therefore:

\[
\bar e
=
\bar L\frac{\partial L}{\partial e}
=
1\cdot(-6)
=
-6.
\]

---

### Slide 21 — Backprop through subtraction

\[
e=a-y.
\]

Local gradients:

\[
\frac{\partial e}{\partial a}=1,
\qquad
\frac{\partial e}{\partial y}=-1.
\]

Given upstream:

\[
\bar e=-6.
\]

Downstream:

\[
\bar a
=
\bar e\cdot1=-6.
\]

\[
\bar y
=
\bar e\cdot(-1)=6.
\]

Usually \(y\) is data, so we do not update it.

---

### Slide 22 — Backprop through addition

\[
a=m+b.
\]

Local gradients:

\[
\frac{\partial a}{\partial m}=1,
\qquad
\frac{\partial a}{\partial b}=1.
\]

Given upstream:

\[
\bar a=-6.
\]

Then:

\[
\bar m=-6,
\qquad
\bar b=-6.
\]

---

### Slide 23 — Backprop through multiplication

\[
m=wx.
\]

Local gradients:

\[
\frac{\partial m}{\partial w}=x=3
\]

\[
\frac{\partial m}{\partial x}=w=2.
\]

Given upstream:

\[
\bar m=-6.
\]

Therefore:

\[
\bar w
=
\bar m x
=
-6\cdot3
=
-18.
\]

\[
\bar x
=
\bar m w
=
-6\cdot2
=
-12.
\]

---

### Slide 24 — Final gradients

\[
\frac{\partial L}{\partial w}=-18
\]

\[
\frac{\partial L}{\partial b}=-6
\]

\[
\frac{\partial L}{\partial x}=-12
\]

\[
\frac{\partial L}{\partial y}=6
\]

For learning, update only parameters:

\[
w\leftarrow w-\eta(-18)
\]

\[
b\leftarrow b-\eta(-6)
\]

---

### Slide 25 — Check with direct calculus

Since:

\[
L=(wx+b-y)^2,
\]

\[
\frac{\partial L}{\partial w}
=
2(wx+b-y)x.
\]

At given values:

\[
2(6+1-10)3
=
2(-3)3
=
-18.
\]

Similarly:

\[
\frac{\partial L}{\partial b}
=
2(wx+b-y)
=
-6.
\]

Backprop matches calculus.

---

### Slide 26 — Same example: one GD step

Let:

\[
\eta=0.1.
\]

Then:

\[
w_{\text{new}}
=
2-0.1(-18)
=
3.8.
\]

\[
b_{\text{new}}
=
1-0.1(-6)
=
1.6.
\]

Old prediction:

\[
\hat y=7.
\]

New prediction:

\[
\hat y=3.8\cdot3+1.6=13.
\]

Overshot.

Message:

> Gradients tell direction; learning rate controls step size.

---

### Slide 27 — Interactive: worked graph

Live lab: `https://nipunbatra.github.io/interactive-articles/autograd/`

Must show:

- forward values;
- upstream gradient at each node;
- local gradient at each edge;
- downstream contribution;
- accumulated gradient;
- one gradient descent step.

Use exact same example:

\[
L=(wx+b-y)^2.
\]

---

# Part IV — From scalar graph to neuron

### Slide 28 — One neuron

\[
z=w^\top x+b
\]

\[
a=\phi(z)
\]

\[
L=\ell(a,y).
\]

Graph:

```text
x,w,b → affine → z → activation → a → loss → L
```

---

### Slide 29 — Backprop through activation

Given:

\[
\bar a=\frac{\partial L}{\partial a},
\]

and

\[
a=\phi(z),
\]

local gradient:

\[
\frac{\partial a}{\partial z}=\phi'(z).
\]

Therefore:

\[
\bar z
=
\bar a\phi'(z).
\]

This \(\bar z\) is often called:

\[
\delta.
\]

---

### Slide 30 — Backprop through affine neuron

\[
z=w^\top x+b.
\]

Given:

\[
\bar z=\delta,
\]

then:

\[
\frac{\partial L}{\partial w}
=
\delta x
\]

\[
\frac{\partial L}{\partial b}
=
\delta
\]

\[
\frac{\partial L}{\partial x}
=
\delta w.
\]

These are just multiplication/addition rules in vector form.

---

### Slide 31 — Sigmoid + BCE

\[
a=\sigma(z)
\]

\[
L=-y\log a-(1-y)\log(1-a)
\]

Result:

\[
\boxed{
\bar z
=
\frac{\partial L}{\partial z}
=
a-y
}
\]

So:

\[
\
abla_w L=(a-y)x.
\]

---

### Slide 32 — Derive sigmoid + BCE simplification

\[
\frac{\partial L}{\partial a}
=
-\frac ya+\frac{1-y}{1-a}
\]

\[
\frac{\partial a}{\partial z}=a(1-a)
\]

\[
\frac{\partial L}{\partial z}
=
\left(
-\frac ya+\frac{1-y}{1-a}
\right)a(1-a)
\]

\[
=
-y(1-a)+(1-y)a
=
a-y.
\]

---

### Slide 33 — Softmax + CE

\[
p=\operatorname{softmax}(z)
\]

\[
L=-\sum_k y_k\log p_k.
\]

Result:

\[
\boxed{
\
abla_zL=p-y
}
\]

Same pattern:

\[
\text{prediction}-\text{target}.
\]

---

### Slide 34 — Interactive: single-neuron backprop

Use the same live autograd lab:
`https://nipunbatra.github.io/interactive-articles/autograd/`

Controls:

- \(x_1,x_2\);
- \(w_1,w_2\);
- \(b\);
- activation: linear/sigmoid/ReLU;
- target \(y\);
- loss: MSE/BCE.

Show:

- \(z,a,L\);
- upstream gradient;
- local gradient;
- \(\delta=\bar z\);
- \(\
abla_w,\
abla_b\).

---

# Part V — Dense layer backprop

### Slide 35 — Dense layer forward

Single example:

\[
z=Wx+b.
\]

Shapes:

\[
x\in\mathbb R^{d_{\text{in}}},
\quad
W\in\mathbb R^{d_{\text{out}}\times d_{\text{in}}},
\quad
b\in\mathbb R^{d_{\text{out}}},
\quad
z\in\mathbb R^{d_{\text{out}}}.
\]

---

### Slide 36 — Dense layer backward

Given:

\[
\bar z=\frac{\partial L}{\partial z}
\in\mathbb R^{d_{\text{out}}},
\]

then:

\[
\
abla_W L=\bar z x^\top
\]

\[
\
abla_b L=\bar z
\]

\[
\bar x=W^\top \bar z.
\]

Interpretation:

- \(\
abla_W\): parameter gradient;
- \(\
abla_b\): parameter gradient;
- \(\bar x\): gradient passed to previous layer.

---

### Slide 37 — Batch version

For batch:

\[
Z=XW^\top+\mathbf 1 b^\top
\]

with:

\[
X:B\times d_{\text{in}},
\quad
W:d_{\text{out}}\times d_{\text{in}},
\quad
Z:B\times d_{\text{out}}.
\]

If:

\[
\bar Z:B\times d_{\text{out}},
\]

then:

\[
\
abla_W=\bar Z^\top X
\]

\[
\
abla_b=\sum_{i=1}^{B}\bar Z_i
\]

\[
\bar X=\bar ZW.
\]

---

### Slide 38 — Shape-checking

\[
\bar Z^\top X:
(d_{\text{out}}\times B)(B\times d_{\text{in}})
=
d_{\text{out}}\times d_{\text{in}}.
\]

Same as:

\[
W.
\]

Message:

\[
\boxed{\text{gradient shape = parameter shape}}
\]

---

# Part VI — Fully worked MLP backprop

### Slide 39 — Two-layer MLP

\[
z^{(1)}=W_1x+b_1
\]

\[
h=\phi(z^{(1)})
\]

\[
z^{(2)}=W_2h+b_2
\]

\[
p=\operatorname{softmax}(z^{(2)})
\]

\[
L=-\log p_y.
\]

---

### Slide 40 — Forward graph

```text
x
↓
affine W1,b1
↓
z1
↓
activation
↓
h
↓
affine W2,b2
↓
z2
↓
softmax + CE
↓
L
```

---

### Slide 41 — Backward: output layer

Softmax + CE gives:

\[
\bar z^{(2)}=p-y.
\]

Then:

\[
\
abla_{W_2}L
=
\bar z^{(2)}h^\top
\]

\[
\
abla_{b_2}L
=
\bar z^{(2)}
\]

\[
\bar h
=
W_2^\top\bar z^{(2)}.
\]

---

### Slide 42 — Backward: hidden activation

Since:

\[
h=\phi(z^{(1)}),
\]

we have:

\[
\bar z^{(1)}
=
\bar h\odot\phi'(z^{(1)}).
\]

This is upstream times local derivative, elementwise.

---

### Slide 43 — Backward: first affine layer

\[
z^{(1)}=W_1x+b_1.
\]

Given:

\[
\bar z^{(1)},
\]

we get:

\[
\
abla_{W_1}L
=
\bar z^{(1)}x^\top
\]

\[
\
abla_{b_1}L
=
\bar z^{(1)}
\]

\[
\bar x
=
W_1^\top\bar z^{(1)}.
\]

---

### Slide 44 — Complete MLP backprop summary

Forward:

\[
z^{(1)}=W_1x+b_1,
\quad
h=\phi(z^{(1)}),
\quad
z^{(2)}=W_2h+b_2,
\quad
p=\operatorname{softmax}(z^{(2)}).
\]

Backward:

\[
\bar z^{(2)}=p-y
\]

\[
\
abla_{W_2}=\bar z^{(2)}h^\top
\]

\[
\bar h=W_2^\top\bar z^{(2)}
\]

\[
\bar z^{(1)}=\bar h\odot\phi'(z^{(1)})
\]

\[
\
abla_{W_1}=\bar z^{(1)}x^\top.
\]

---

### Slide 45 — Interactive: MLP backprop

Use the same live autograd lab:
`https://nipunbatra.github.io/interactive-articles/autograd/`

Network:

\[
2\rightarrow3\rightarrow2.
\]

Show:

- forward activations;
- softmax probabilities;
- loss;
- \(\bar z^{(2)}\);
- \(\bar h\);
- \(\bar z^{(1)}\);
- parameter gradients;
- one update step.

---

# Part VII — Autodiff

### Slide 46 — Backprop is reverse-mode autodiff

\[
\boxed{
\text{backpropagation}
=
\text{reverse-mode automatic differentiation}
}
\]

It is not:

- symbolic differentiation;
- finite differences;
- hand-derived closed form.

It is chain rule over the executed computation graph.

---

### Slide 47 — Why reverse mode?

Neural network training has:

\[
f:\mathbb R^P\rightarrow\mathbb R
\]

where:

- input: all parameters \(\theta\in\mathbb R^P\);
- output: scalar loss \(L\).

Reverse mode efficiently computes:

\[
\
abla_\theta L
\]

in roughly a small constant multiple of forward-pass cost.

---

### Slide 48 — What PyTorch/JAX do

1. run forward pass;
2. record operations;
3. store needed values;
4. start backward with \(\bar L=1\);
5. apply local backward rules;
6. accumulate gradients;
7. expose `.grad`.

---

### Slide 49 — PyTorch scalar demo

Canonical notebook: `notebooks/L03/00_backprop_autograd_complete.ipynb`

Use checkpoints 3–8 for the complete scalar ledger, symbolic and finite-
difference checks, retained intermediate gradients, and `.grad` accumulation.

```python
import torch

x = torch.tensor(3.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(10.0)

m = w * x
a = m + b
e = a - y
L = e ** 2

L.backward()

print(L.item())
print(w.grad, b.grad)
```

Expected:

\[
L=9,\qquad
w.grad=-18,\qquad
b.grad=-6.
\]

---

### Slide 50 — Gradients accumulate in frameworks

If you call:

```python
L.backward()
L.backward()
```

then gradients double.

Training loop:

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Conceptual reason:

\[
\bar x \leftarrow \bar x+\text{new contribution}.
\]

---

# Part VIII — Gradient checking and gradient flow

### Slide 51 — Finite difference check

\[
\frac{\partial L}{\partial\theta_j}
\approx
\frac{
L(\theta+\epsilon e_j)-L(\theta-\epsilon e_j)
}{2\epsilon}.
\]

Useful for debugging custom layers.

Not useful for training:

\[
2P\text{ forward passes for }P\text{ parameters.}
\]

---

### Slide 52 — Gradient flow in deep networks

For a deep chain:

\[
h_L=f_L(f_{L-1}(\cdots f_1(x))).
\]

Gradient:

\[
\frac{\partial L}{\partial h_0}
=
\frac{\partial L}{\partial h_L}
\prod_{\ell=1}^{L}
\frac{\partial h_\ell}{\partial h_{\ell-1}}.
\]

Products can vanish or explode.

---

### Slide 53 — Sigmoid saturation

\[
\sigma'(z)=\sigma(z)(1-\sigma(z))
\]

\[
0<\sigma'(z)\le \frac14.
\]

For large \(|z|\):

\[
\sigma'(z)\approx0.
\]

Deep sigmoid networks can suffer vanishing gradients.

---

### Slide 54 — Interactive: gradient flow

Live lab: `https://nipunbatra.github.io/interactive-articles/vanishing-gradients/`

Controls:

- depth;
- activation: sigmoid/tanh/ReLU;
- weight scale;
- input scale.

Show:

- activations layer-wise;
- gradient magnitudes layer-wise;
- log-gradient plot.

Question:

> Where do gradients vanish or explode?

---

# Part IX — Summary

### Slide 55 — Backprop in one sentence

\[
\boxed{
\text{Backprop computes gradients by passing sensitivities backward through a graph.}
}
\]

At each node:

\[
\boxed{
\text{downstream}
=
\text{upstream}
\times
\text{local}
}
\]

At branches:

\[
\boxed{
\text{gradients add}
}
\]

---

### Slide 56 — Key formulas

Scalar node:

\[
\bar u=\bar v\frac{\partial v}{\partial u}
\]

Dense layer:

\[
z=Wx+b
\]

\[
\
abla_W=\bar z x^\top
\]

\[
\
abla_b=\bar z
\]

\[
\bar x=W^\top\bar z
\]

Softmax CE:

\[
\bar z=p-y.
\]

---

### Slide 57 — Next lecture

Now we can compute:

\[
\
abla_\theta L.
\]

Next:

\[
\theta_{t+1}=\theta_t-\eta\
abla_\theta L.
\]

Topics:

- SGD;
- minibatches;
- momentum;
- Adam;
- learning-rate schedules;
- optimization pathologies.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Motivation | 1–3 | 4 min |
| Backprop vocabulary | 4–11 | 11 min |
| Gradient accumulation | 12–16 | 8 min |
| Fully worked scalar example | 17–27 | 16 min |
| One neuron | 28–34 | 10 min |
| Dense layer | 35–38 | 7 min |
| Two-layer MLP | 39–45 | 12 min |
| Autodiff | 46–50 | 7 min |
| Gradient check/flow | 51–54 | 4 min |
| Summary | 55–57 | 3 min |

Skim Slides **37, 51, 53** if short.

---

# Interactives

1. [Autograd lab](https://nipunbatra.github.io/interactive-articles/autograd/)
   The main companion: step through the scalar graph, branch accumulation,
   local derivative rules, and small neural-network modules.

2. [Vanishing-gradients lab](https://nipunbatra.github.io/interactive-articles/vanishing-gradients/)
   Compare gradient flow through deep sigmoid, tanh, and ReLU chains.

---

# Canonical notebook

Use one student-facing notebook throughout:
`notebooks/L03/00_backprop_autograd_complete.ipynb`.

Its checkpoints follow the lecture in order: local rules; manual scalar graph;
symbolic and finite-difference checks; PyTorch autograd; branch accumulation;
one vector neuron; dense VJP; three-example mean batch; microbatch equivalence;
one optimizer step; and a tiny dense–ReLU–dense MLP. The older focused
notebooks remain in the repository only for backward-compatible links.

---

# Optional mini-deck: derivative, gradient, Jacobian, Hessian

Use as appendix or separate 25-minute revision.

## A1 — Derivative

For scalar function:

\[
f:\mathbb R\rightarrow\mathbb R
\]

\[
f'(x)=\frac{df}{dx}
=
\lim_{\epsilon\to0}
\frac{f(x+\epsilon)-f(x)}{\epsilon}.
\]

Meaning: local slope.

---

## A2 — Partial derivative

For:

\[
f:\mathbb R^d\rightarrow\mathbb R
\]

\[
\frac{\partial f}{\partial x_j}
\]

changes only \(x_j\), holds others fixed.

---

## A3 — Gradient

\[
\
abla_x f
=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\vdots\\
\frac{\partial f}{\partial x_d}
\end{bmatrix}.
\]

Local approximation:

\[
f(x+\Delta x)
\approx
f(x)+\
abla f(x)^\top\Delta x.
\]

---

## A4 — Jacobian

For vector function:

\[
f:\mathbb R^n\rightarrow\mathbb R^m
\]

\[
J_{ij}
=
\frac{\partial f_i}{\partial x_j}.
\]

So:

\[
J\in\mathbb R^{m\times n}.
\]

---

## A5 — Hessian

For scalar function:

\[
f:\mathbb R^n\rightarrow\mathbb R
\]

\[
H_{ij}
=
\frac{\partial^2 f}{\partial x_i\partial x_j}.
\]

Second-order local approximation:

\[
f(x+\Delta x)
\approx
f(x)+\
abla f^\top\Delta x
+
\frac12\Delta x^\top H\Delta x.
\]

---

## A6 — In neural networks, what do we usually need?

Training loss:

\[
L:\mathbb R^P\rightarrow\mathbb R.
\]

Need:

\[
\
abla_\theta L\in\mathbb R^P.
\]

Usually do not explicitly form:

- full Jacobians;
- full Hessians.

Backprop efficiently computes vector-Jacobian products.
