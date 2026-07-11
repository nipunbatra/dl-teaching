---
title: "Lecture 6: Making Deep Networks Trainable"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 6: Making Deep Networks Trainable

## Initialization, Activations, Normalization and Residual Connections

Target: **80 minutes, 55–60 slides**

---

# Part I — Why depth breaks naive networks

### Slide 1 — Title

**Making Deep Networks Trainable**

> Initialization, activations, normalization and residual connections

Visual: a deep network with signals travelling forward and gradients travelling backward.

---

### Slide 2 — We already have all the ingredients… apparently

From earlier lectures:

\[
h^{(\ell)}
=
\phi\!\left(W_\ell h^{(\ell-1)}+b_\ell\right)
\]

\[
\
abla_\theta L
\quad\text{via backpropagation}
\]

\[
\theta\leftarrow\theta-\eta\
abla_\theta L
\]

Question:

> Why not simply stack 100 layers and train?

---

### Slide 3 — The deep-network experiment

Construct a randomly initialized 100-layer MLP:

\[
h^{(\ell)}
=
\phi(W_\ell h^{(\ell-1)}).
\]

Track two quantities:

\[
q_\ell
=
\operatorname{Var}\left(h^{(\ell)}\right)
\]

and

\[
g_\ell
=
\left\|
\frac{\partial L}{\partial h^{(\ell)}}
\right\|_2.
\]

Possible outcomes:

\[
q_\ell,g_\ell\rightarrow0
\]

or

\[
q_\ell,g_\ell\rightarrow\infty.
\]

---

### Slide 4 — Four ways training can fail

1. **Vanishing activations**
   \[
   h^{(\ell)}\approx0
   \]

2. **Exploding activations**
   \[
   \|h^{(\ell)}\|\gg1
   \]

3. **Vanishing gradients**
   \[
   \left\|\
abla_{W_1}L\right\|\approx0
   \]

4. **Exploding gradients**
   \[
   \left\|\
abla_{W_1}L\right\|\gg1
   \]

Use four layer-wise plots.

---

### Slide 5 — Forward and backward recurrences

Forward:

\[
z^{(\ell)}=W_\ell h^{(\ell-1)}
\]

\[
h^{(\ell)}=\phi(z^{(\ell)}).
\]

Backward:

\[
\bar h^{(\ell-1)}
=
W_\ell^\top
\left(
\bar h^{(\ell)}
\odot
\phi'(z^{(\ell)})
\right).
\]

Each layer transforms:

- the forward signal;
- the backward gradient.

---

### Slide 6 — Product-of-Jacobians view

For a deep composition:

\[
h^{(L)}
=
f_L\circ f_{L-1}\circ\cdots\circ f_1(x),
\]

the gradient contains:

\[
\frac{\partial h^{(L)}}{\partial x}
=
J_LJ_{L-1}\cdots J_1.
\]

If typical singular values of \(J_\ell\) are:

\[
<1 \Rightarrow \text{vanishing}
\]

\[
>1 \Rightarrow \text{exploding}.
\]

---

### Slide 7 — Today’s four interventions

```text
Problem                      Intervention
────────────────────────────────────────────
Bad initial scale         →  Initialization
Poor local derivatives    →  Activations
Drifting/unstable scale   →  Normalization
Very long gradient paths  →  Residual connections
```

---

# Part II — Activation functions and gradient flow

### Slide 8 — What an activation must do

An activation should ideally:

- introduce nonlinearity;
- preserve useful gradient;
- be computationally simple;
- avoid pathological output distributions.

Without it:

\[
W_2(W_1x)=W'x.
\]

Depth collapses to one affine transformation.

---

### Slide 9 — Sigmoid

\[
\sigma(z)
=
\frac{1}{1+e^{-z}}
\]

\[
\sigma'(z)
=
\sigma(z)(1-\sigma(z)).
\]

Properties:

\[
0<\sigma(z)<1
\]

\[
0<\sigma'(z)\le\frac14.
\]

Visual: function and derivative beneath it.

---

### Slide 10 — Sigmoid saturation

For large \(|z|\):

\[
\sigma'(z)\approx0.
\]

Through \(L\) sigmoid layers:

\[
\prod_{\ell=1}^{L}\sigma'(z_\ell)
\le
\left(\frac14\right)^L.
\]

For \(L=10\):

\[
\left(\frac14\right)^{10}
\approx9.5\times10^{-7}.
\]

Message:

> Even before multiplying by weights, the gradient may collapse.

Glorot and Bengio specifically identified saturation and poor signal propagation as key difficulties in training deep feedforward networks.

---

### Slide 11 — Sigmoid is not zero-centred

\[
\sigma(z)\in(0,1).
\]

Therefore hidden activations have positive mean.

For:

\[
z_j^{(\ell+1)}
=
\sum_i w_{ji}^{(\ell+1)}h_i^{(\ell)},
\]

positive-mean activations can produce correlated parameter-gradient directions.

Use a visual comparing zero-centred and positive-only point clouds.

---

### Slide 12 — Tanh

\[
\tanh(z)
=
\frac{e^z-e^{-z}}{e^z+e^{-z}}
\]

\[
\tanh'(z)
=
1-\tanh^2(z).
\]

Advantages over sigmoid:

- outputs are approximately zero-centred;
- maximum derivative is \(1\).

But:

\[
|z|\gg1
\Rightarrow
\tanh'(z)\approx0.
\]

Still saturates.

---

### Slide 13 — ReLU

\[
\operatorname{ReLU}(z)=\max(0,z)
\]

\[
\operatorname{ReLU}'(z)
=
\begin{cases}
1,&z>0,\\
0,&z<0.
\end{cases}
\]

Advantages:

- no positive-side saturation;
- sparse activations;
- simple derivative.

---

### Slide 14 — Dead ReLUs

A ReLU unit receiving:

\[
z<0
\]

has:

\[
h=0,\qquad \frac{\partial h}{\partial z}=0.
\]

If parameter updates keep it negative for all data:

\[
\text{the neuron may stop learning.}
\]

Possible causes:

- large learning rate;
- poor initialization;
- large negative bias.

---

### Slide 15 — Leaky ReLU and PReLU

Leaky ReLU:

\[
\phi(z)=
\begin{cases}
z,&z>0,\\
\alpha z,&z\le0.
\end{cases}
\]

PReLU learns \(\alpha\).

\[
\phi'(z)=
\begin{cases}
1,&z>0,\\
\alpha,&z\le0.
\end{cases}
\]

He et al. proposed PReLU and an initialization designed for rectifier nonlinearities.

---

### Slide 16 — Smooth modern activations

GELU:

\[
\operatorname{GELU}(z)=z\Phi(z)
\]

SiLU/Swish:

\[
\operatorname{SiLU}(z)=z\sigma(z).
\]

Contrast:

- ReLU: hard gate;
- GELU: smooth probabilistic-looking gate;
- SiLU: smooth self-gating.

Do not derive these deeply; show their shapes.

---

### Slide 17 — Activation comparison

| Activation | Zero-centred | Saturates | Negative gradient | Typical use |
|---|---:|---:|---:|---|
| sigmoid | no | both sides | small | output probabilities, gates |
| tanh | yes | both sides | small | recurrent states, older MLPs |
| ReLU | partly | negative side | zero | CNNs, MLPs |
| Leaky ReLU | partly | no hard death | nonzero | vision variants |
| GELU/SiLU | partly | smooth | nonzero/small | Transformers, modern networks |

---

### Slide 18 — Interactive 1: activation laboratory

`activation-lab.html`

Controls:

- sigmoid/tanh/ReLU/leaky ReLU/GELU/SiLU;
- input mean;
- input variance;
- number of repeated layers;
- weight scale.

Show:

1. activation function;
2. derivative;
3. activation histogram;
4. gradient magnitude across layers;
5. percentage of saturated/dead units.

---

# Part III — Why initialization matters

### Slide 19 — Why not initialize all weights to zero?

Suppose two hidden neurons begin identically:

\[
w_1=w_2=0.
\]

They receive identical:

- activations;
- gradients;
- updates.

Therefore:

\[
w_1^{(t)}=w_2^{(t)}
\quad\forall t.
\]

They learn the same feature.

\[
\boxed{\text{Random initialization breaks symmetry.}}
\]

Biases can generally start at zero; hidden weights should not all be equal.

---

### Slide 20 — A single linear neuron

\[
z
=
\sum_{i=1}^{n}w_ix_i.
\]

Assume:

\[
\mathbb E[x_i]=0,
\qquad
\mathbb E[w_i]=0,
\]

and independence.

Then:

\[
\mathbb E[z]=0.
\]

What is:

\[
\operatorname{Var}(z)?
\]

---

### Slide 21 — Forward-variance derivation

\[
\operatorname{Var}(z)
=
\operatorname{Var}
\left(
\sum_{i=1}^{n}w_ix_i
\right).
\]

Assuming independence:

\[
\operatorname{Var}(z)
=
\sum_{i=1}^{n}
\operatorname{Var}(w_ix_i).
\]

For independent zero-mean \(w_i,x_i\):

\[
\operatorname{Var}(w_ix_i)
=
\operatorname{Var}(w_i)\operatorname{Var}(x_i).
\]

Hence:

\[
\boxed{
\operatorname{Var}(z)
=
n\operatorname{Var}(w)\operatorname{Var}(x)
}
\]

---

### Slide 22 — Variance-preservation condition

We want:

\[
\operatorname{Var}(z)
\approx
\operatorname{Var}(x).
\]

Thus:

\[
n\operatorname{Var}(w)\approx1.
\]

Therefore:

\[
\boxed{
\operatorname{Var}(w)
\approx
\frac1{n_{\text{in}}}
}
\]

or:

\[
\operatorname{Std}(w)
\approx
\frac1{\sqrt{n_{\text{in}}}}.
\]

---

### Slide 23 — What if the scale is wrong?

If:

\[
n\operatorname{Var}(w)=c,
\]

after \(L\) linear layers:

\[
\operatorname{Var}(h^{(L)})
\approx
c^L\operatorname{Var}(x).
\]

Examples:

\[
c=0.5,\ L=20
\Rightarrow
c^L\approx9.5\times10^{-7}
\]

\[
c=2,\ L=20
\Rightarrow
c^L\approx10^6.
\]

Small per-layer mismatch compounds exponentially.

---

### Slide 24 — Backward variance matters too

For a linear layer:

\[
z=Wx.
\]

Backward:

\[
\bar x=W^\top\bar z.
\]

Analogously:

\[
\operatorname{Var}(\bar x)
\approx
n_{\text{out}}
\operatorname{Var}(w)
\operatorname{Var}(\bar z).
\]

Forward preservation suggests:

\[
\operatorname{Var}(w)\approx\frac1{n_{\text{in}}}.
\]

Backward preservation suggests:

\[
\operatorname{Var}(w)\approx\frac1{n_{\text{out}}}.
\]

---

### Slide 25 — Xavier/Glorot initialization

Compromise between forward and backward preservation:

\[
\boxed{
\operatorname{Var}(w)
=
\frac{2}{n_{\text{in}}+n_{\text{out}}}
}
\]

Normal form:

\[
w_{ij}
\sim
\mathcal N
\left(
0,
\frac{2}{n_{\text{in}}+n_{\text{out}}}
\right).
\]

Uniform form:

\[
w_{ij}
\sim
\mathcal U
\left(
-\sqrt{\frac6{n_{\text{in}}+n_{\text{out}}}},
\sqrt{\frac6{n_{\text{in}}+n_{\text{out}}}}
\right).
\]

Glorot initialization was motivated by preserving useful signal and gradient scales across layers.

---

### Slide 26 — Why ReLU changes the calculation

Suppose:

\[
z
\]

is zero-mean and symmetric.

For:

\[
h=\operatorname{ReLU}(z),
\]

approximately half the inputs are set to zero.

At the second-moment level:

\[
\mathbb E[h^2]
\approx
\frac12\mathbb E[z^2].
\]

So ReLU removes approximately half the signal energy.

---

### Slide 27 — He/Kaiming initialization

To compensate for ReLU’s gating:

\[
\frac12n_{\text{in}}\operatorname{Var}(w)\approx1.
\]

Therefore:

\[
\boxed{
\operatorname{Var}(w)
\approx
\frac2{n_{\text{in}}}
}
\]

and:

\[
\operatorname{Std}(w)
\approx
\sqrt{\frac2{n_{\text{in}}}}.
\]

This rectifier-aware initialization was derived by He et al. and enabled substantially deeper rectified networks to be trained from scratch.

---

### Slide 28 — Generalized gain

For leaky ReLU with negative slope \(a\):

\[
\phi(z)
=
\begin{cases}
z,&z>0,\\
az,&z\le0,
\end{cases}
\]

approximately:

\[
\mathbb E[\phi(z)^2]
=
\frac{1+a^2}{2}\mathbb E[z^2].
\]

Thus:

\[
\boxed{
\operatorname{Var}(w)
\approx
\frac{2}{(1+a^2)n_{\text{in}}}
}
\]

---

### Slide 29 — Fully worked initialization example

Consider:

\[
n_{\text{in}}=100,
\qquad
x_i\sim(0,1).
\]

#### Case A: \(\operatorname{Std}(w)=0.01\)

\[
\operatorname{Var}(w)=10^{-4}
\]

\[
\operatorname{Var}(z)
=
100(10^{-4})(1)
=
0.01.
\]

Signal shrinks by factor \(100\) in variance per layer.

#### Case B: \(\operatorname{Std}(w)=1\)

\[
\operatorname{Var}(z)=100.
\]

Signal explodes.

#### Case C: Xavier with equal fan-in/fan-out

\[
\operatorname{Var}(w)=\frac1{100}
\]

\[
\operatorname{Var}(z)=1.
\]

#### Case D: He + ReLU

\[
\operatorname{Var}(w)=\frac2{100}.
\]

Before ReLU:

\[
\operatorname{Var}(z)\approx2.
\]

After ReLU, second moment is approximately restored:

\[
\mathbb E[h^2]\approx1.
\]

---

### Slide 30 — Convolutional fan-in

For a convolution with:

- kernel \(k_h\times k_w\);
- \(C_{\text{in}}\) input channels,

\[
n_{\text{in}}
=
k_hk_wC_{\text{in}}.
\]

For a \(3\times3\) convolution with \(64\) channels:

\[
n_{\text{in}}
=
3\cdot3\cdot64
=
576.
\]

He standard deviation:

\[
\sqrt{\frac2{576}}
\approx0.0589.
\]

---

### Slide 31 — Initialization is not a guarantee

Variance preservation is based on approximations:

- independence;
- zero mean;
- equal variance;
- wide layers;
- idealized input distributions.

During training:

- weights become correlated;
- means shift;
- representations change.

Initialization makes training start in a sensible regime; it does not keep it there.

---

### Slide 32 — Interactive 2: signal propagation

`initialization-signal-flow.html`

Controls:

- depth \(1\)–\(200\);
- width;
- activation;
- initialization:
  - constant;
  - tiny Gaussian;
  - standard Gaussian;
  - Xavier;
  - He;
- input mean and variance.

Plots by layer:

\[
\operatorname{mean}(h^{(\ell)})
\]

\[
\operatorname{std}(h^{(\ell)})
\]

\[
\operatorname{std}(z^{(\ell)})
\]

\[
\left\|\bar h^{(\ell)}\right\|.
\]

---

# Part IV — Normalization

### Slide 33 — Initialization only controls time \(t=0\)

After several updates:

\[
W_\ell^{(t)}
\
eq
W_\ell^{(0)}.
\]

Activation distributions may change:

\[
\mu_\ell^{(t)},\quad
\sigma_\ell^{(t)}.
\]

Question:

> Can the network explicitly control intermediate scales?

---

### Slide 34 — Basic normalization operation

Given activations \(x\):

\[
\mu
=
\frac1m\sum_{i=1}^{m}x_i
\]

\[
\sigma^2
=
\frac1m\sum_{i=1}^{m}(x_i-\mu)^2
\]

\[
\hat x_i
=
\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
\]

This produces approximately:

\[
\operatorname{mean}(\hat x)=0,
\qquad
\operatorname{var}(\hat x)=1.
\]

---

### Slide 35 — Why add \(\gamma\) and \(\beta\)?

Normalization alone would restrict representation scale.

Instead use:

\[
y_i
=
\gamma\hat x_i+\beta.
\]

The model can learn:

- desired scale through \(\gamma\);
- desired shift through \(\beta\).

If helpful, it can recover identity-like transformations.

---

### Slide 36 — The crucial question: normalize over which axes?

Suppose:

\[
X\in\mathbb R^{B\times D}.
\]

Possible choices:

- across batch \(B\);
- across features \(D\);
- across spatial dimensions;
- across channel groups.

The normalization formula is similar; the axes differ.

---

### Slide 37 — Batch Normalization for an MLP

For each feature \(j\):

\[
\mu_j
=
\frac1B\sum_{i=1}^{B}x_{ij}
\]

\[
\sigma_j^2
=
\frac1B\sum_{i=1}^{B}(x_{ij}-\mu_j)^2.
\]

Then:

\[
\hat x_{ij}
=
\frac{x_{ij}-\mu_j}
{\sqrt{\sigma_j^2+\epsilon}}.
\]

Statistics are shared across examples in the minibatch.

---

### Slide 38 — BatchNorm axes diagram

For matrix \(B\times D\):

```text
             Features
          1   2   3   4
Batch 1   ●   ●   ●   ●
      2   ●   ●   ●   ●
      3   ●   ●   ●   ●
      4   ●   ●   ●   ●
```

BatchNorm computes one mean/variance **down each feature column**.

Use colour-coded columns.

---

### Slide 39 — BatchNorm for convolution

For:

\[
X\in\mathbb R^{B\times C\times H\times W},
\]

BatchNorm typically computes one pair of statistics per channel:

\[
\mu_c,\sigma_c^2
\]

over:

\[
B,H,W.
\]

Thus each channel has learned:

\[
\gamma_c,\beta_c.
\]

---

### Slide 40 — Training versus inference

During training:

\[
\mu_{\mathcal B},
\quad
\sigma_{\mathcal B}^2
\]

come from the current minibatch.

Maintain running estimates:

\[
\mu_{\text{run}}
\leftarrow
\rho\mu_{\text{run}}
+
(1-\rho)\mu_{\mathcal B}.
\]

At inference use:

\[
\mu_{\text{run}},
\quad
\sigma_{\text{run}}^2.
\]

Important practical distinction:

```python
model.train()
model.eval()
```

---

### Slide 41 — Worked BatchNorm example

Batch values for one feature:

\[
x=[1,2,3,4].
\]

Mean:

\[
\mu=\frac{1+2+3+4}{4}=2.5.
\]

Variance:

\[
\sigma^2
=
\frac{
(1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2
}{4}
=
1.25.
\]

Ignoring \(\epsilon\):

\[
\hat x
\approx
[-1.342,-0.447,0.447,1.342].
\]

With:

\[
\gamma=2,\qquad \beta=1,
\]

\[
y
\approx
[-1.683,0.106,1.894,3.683].
\]

---

### Slide 42 — Why does BatchNorm help?

Original motivation:

> reduce changes in intermediate input distributions.

Empirically, BatchNorm enabled higher learning rates and substantially faster training in the original work.

But avoid teaching “internal covariate shift” as the settled explanation. Later work found that reduction of such distribution shift was not necessary to explain BatchNorm’s success and instead emphasized smoother, more predictable optimization behaviour.

Best teaching statement:

\[
\boxed{
\text{BatchNorm reparameterizes the network in a way that often stabilizes optimization.}
}
\]

---

### Slide 43 — BatchNorm limitations

- Depends on minibatch statistics.
- Very small batches give noisy estimates.
- Training and inference computations differ.
- An example’s output depends on other examples in its batch.
- Less natural for autoregressive and variable-length sequence models.

---

### Slide 44 — Layer Normalization

For each example \(i\), normalize across its \(D\) features:

\[
\mu_i
=
\frac1D\sum_{j=1}^{D}x_{ij}
\]

\[
\sigma_i^2
=
\frac1D\sum_{j=1}^{D}(x_{ij}-\mu_i)^2.
\]

Then:

\[
\hat x_{ij}
=
\frac{x_{ij}-\mu_i}
{\sqrt{\sigma_i^2+\epsilon}}.
\]

LayerNorm uses the same computation during training and inference and does not depend on batch size.

---

### Slide 45 — LayerNorm axes diagram

For \(B\times D\):

```text
             Features
          1   2   3   4
Batch 1   ●───●───●───●
      2   ●───●───●───●
      3   ●───●───●───●
      4   ●───●───●───●
```

LayerNorm computes one mean/variance **across each row**.

---

### Slide 46 — BatchNorm versus LayerNorm

| Property | BatchNorm | LayerNorm |
|---|---|---|
| Statistics over | batch/examples | features of one example |
| Batch-size dependent | yes | no |
| Train/inference difference | yes | no |
| Common use | CNNs | Transformers, sequence models |
| Adds batch noise | yes | no |
| Natural for variable lengths | less | yes |

---

### Slide 47 — Optional: RMSNorm

RMSNorm omits mean subtraction:

\[
\operatorname{RMS}(x)
=
\sqrt{
\frac1D\sum_jx_j^2+\epsilon
}
\]

\[
y
=
\gamma
\frac{x}{\operatorname{RMS}(x)}.
\]

Teaching point:

> Some architectures need scale control more than exact centring.

Mark as optional/modern extension.

---

### Slide 48 — Interactive 3: normalization axes

`normalization-axes.html`

Input matrix:

\[
X\in\mathbb R^{B\times D}.
\]

Toggle:

- no normalization;
- BatchNorm;
- LayerNorm;
- RMSNorm.

Show:

- selected reduction axes;
- means and variances;
- normalized matrix;
- effect of changing one example;
- effect of changing batch size.

---

### Slide 49 — Interactive 4: BatchNorm train versus inference

`batchnorm-train-test.html`

Controls:

- minibatch size;
- data distribution;
- running-momentum coefficient;
- distribution shift;
- train/eval mode.

Show:

- batch statistics;
- running statistics;
- normalized outputs;
- mismatch caused by poor running estimates.

---

# Part V — Residual connections

### Slide 50 — The degradation problem

Suppose a 20-layer network trains well.

A 56-layer network has strictly more representational capacity: it could copy the first 20 layers and make the rest identities.

Yet historically, deeper plain networks could obtain **higher training error**, not merely higher test error.

That indicates an optimization problem. The original ResNet work introduced residual learning specifically to make much deeper networks easier to optimize.

---

### Slide 51 — Learn the residual instead

Desired mapping:

\[
H(x).
\]

Instead of learning \(H\) directly, learn:

\[
F(x)=H(x)-x.
\]

Then:

\[
\boxed{
H(x)=x+F(x)
}
\]

Residual block:

\[
x_{\ell+1}=x_\ell+F_\ell(x_\ell).
\]

---

### Slide 52 — Residual block diagram

```text
                   ┌───────────────────────┐
                   │                       │
x ──→ layer ─→ act ─→ layer ───────────── + ─→ output
│                                          ↑
└──────────────── identity shortcut ───────┘
```

Label:

- residual branch: \(F(x)\);
- shortcut branch: \(x\);
- output: \(x+F(x)\).

---

### Slide 53 — Why identity is easy

If the optimal transformation is close to identity:

\[
H(x)\approx x,
\]

then the residual should be small:

\[
F(x)\approx0.
\]

It can be easier to learn a correction to \(x\) than reconstruct \(x\) through many nonlinear layers.

---

### Slide 54 — Forward information path

Repeated residual blocks:

\[
x_{\ell+1}=x_\ell+F_\ell(x_\ell).
\]

Expanding:

\[
x_{\ell+2}
=
x_\ell+F_\ell(x_\ell)+F_{\ell+1}(x_{\ell+1}).
\]

In general:

\[
\boxed{
x_L
=
x_\ell
+
\sum_{i=\ell}^{L-1}F_i(x_i)
}
\]

There is an explicit direct contribution from \(x_\ell\) to \(x_L\).

---

### Slide 55 — Backward gradient path

For one block:

\[
x_{\ell+1}=x_\ell+F_\ell(x_\ell).
\]

Then:

\[
\frac{\partial L}{\partial x_\ell}
=
\frac{\partial L}{\partial x_{\ell+1}}
\left(
I+
\frac{\partial F_\ell}{\partial x_\ell}
\right).
\]

Expanded:

\[
\frac{\partial L}{\partial x_\ell}
=
\underbrace{
\frac{\partial L}{\partial x_{\ell+1}}
}_{\text{identity path}}
+
\underbrace{
\frac{\partial L}{\partial x_{\ell+1}}
\frac{\partial F_\ell}{\partial x_\ell}
}_{\text{residual path}}.
\]

The identity term does not require multiplication through the residual branch.

Identity-mapping analyses of ResNets emphasize this direct propagation of forward and backward signals.

---

### Slide 56 — Worked scalar residual example

Let:

\[
y=x+F(x),
\qquad
F(x)=wx^2.
\]

Then:

\[
y=x+wx^2.
\]

Derivative:

\[
\frac{dy}{dx}
=
1+2wx.
\]

Take:

\[
w=0.1,\qquad x=2.
\]

Then:

\[
\frac{dy}{dx}
=
1+0.4
=
1.4.
\]

Without shortcut:

\[
y=wx^2
\Rightarrow
\frac{dy}{dx}=0.4.
\]

The residual block retains the identity contribution \(1\).

---

### Slide 57 — Dimension-changing shortcuts

Addition requires matching shapes:

\[
x\in\mathbb R^{d_{\text{in}}},
\qquad
F(x)\in\mathbb R^{d_{\text{out}}}.
\]

If dimensions differ, use projection:

\[
x_{\ell+1}
=
P_\ell x_\ell+F_\ell(x_\ell),
\]

where:

\[
P_\ell\in
\mathbb R^{d_{\text{out}}\times d_{\text{in}}}.
\]

In CNNs, this may be a \(1\times1\) convolution.

---

### Slide 58 — Post-activation versus pre-activation

Original-style block:

```text
Linear/Conv → Norm → ReLU → Linear/Conv → Norm → Add → ReLU
```

Pre-activation block:

```text
Norm → ReLU → Linear/Conv → Norm → ReLU → Linear/Conv → Add
```

Pre-activation keeps the shortcut path closer to a clean identity, which was shown to improve optimization in very deep residual networks.

---

### Slide 59 — Residuals do not magically solve everything

Residual networks still need attention to:

- branch scale;
- initialization;
- normalization placement;
- learning rate;
- depth-dependent accumulation.

Residual networks can even be trained without normalization using specially designed initialization such as Fixup, illustrating that initialization, normalization and residual structure interact rather than operate independently.

---

### Slide 60 — Interactive 5: residual gradient flow

`residual-gradient-flow.html`

Compare:

Plain:

\[
x_{\ell+1}=\phi(W_\ell x_\ell)
\]

Residual:

\[
x_{\ell+1}=x_\ell+\alpha F_\ell(x_\ell).
\]

Controls:

- depth;
- weight scale;
- activation;
- residual scale \(\alpha\);
- normalization on/off;
- pre-activation/post-activation.

Show layer-wise:

\[
\operatorname{Var}(x_\ell)
\]

\[
\left\|
\frac{\partial L}{\partial x_\ell}
\right\|
\]

\[
\left\|
J_{\ell:1}
\right\|.
\]

---

# Part VI — Putting the pieces together

### Slide 61 — One 50-layer ablation

Train the same network under six configurations:

| Configuration | Expected behaviour |
|---|---|
| sigmoid + standard normal | saturation, unstable |
| sigmoid + Xavier | improved but gradients may vanish |
| ReLU + Xavier | signal tends to shrink |
| ReLU + He | stable initial scale |
| ReLU + He + normalization | more robust optimization |
| ReLU + He + normalization + residuals | deepest and easiest to train |

Use train-loss and gradient-norm plots.

---

### Slide 62 — What each technique changes

\[
\boxed{
\text{Initialization}
\Rightarrow
\text{starting distribution of signals}
}
\]

\[
\boxed{
\text{Activation}
\Rightarrow
\text{local nonlinearity and local derivative}
}
\]

\[
\boxed{
\text{Normalization}
\Rightarrow
\text{scale and parameterization during training}
}
\]

\[
\boxed{
\text{Residual connection}
\Rightarrow
\text{path length for information and gradients}
}
\]

---

### Slide 63 — These are coupled choices

Examples:

- ReLU motivates He rather than Xavier initialization.
- BatchNorm reduces sensitivity to initialization but does not make initialization irrelevant.
- Residual branches may need smaller initial scale.
- LayerNorm placement affects residual-stream stability.
- Learning rate that works with normalization may fail without it.

Do not teach them as isolated “tricks.”

---

### Slide 64 — Diagnostic dashboard

For each layer, log:

#### Activations

\[
\operatorname{mean}(h_\ell),
\quad
\operatorname{std}(h_\ell),
\quad
\min h_\ell,
\quad
\max h_\ell.
\]

#### Gradients

\[
\|\
abla_{W_\ell}L\|_2
\]

\[
\frac{\|\
abla_{W_\ell}L\|_2}
{\|W_\ell\|_2+\epsilon}.
\]

#### Activation health

- percentage zero;
- percentage saturated;
- percentage non-finite.

---

### Slide 65 — Reading common symptoms

| Observation | Likely issue |
|---|---|
| activation std decreases with depth | initialization too small or saturating activation |
| activation std explodes | weight scale too large |
| early-layer gradients near zero | vanishing gradient |
| many ReLUs always zero | dead units |
| train loss becomes NaN | exploding activations/gradients or excessive LR |
| BatchNorm poor at inference | bad running statistics or missing `eval()` |
| deep plain network underfits | optimization degradation |

---

### Slide 66 — Practical recipes

#### Moderate-depth MLP

\[
\text{ReLU/GELU + He-like initialization}
\]

Add LayerNorm if training is unstable.

#### CNN

\[
\text{He init + ReLU/SiLU + BatchNorm + residual blocks}
\]

#### Transformer-like model

\[
\text{LayerNorm/RMSNorm + residual stream + GELU/SiLU}
\]

with carefully scaled initialization and learning-rate warmup.

---

### Slide 67 — What not to overclaim

Do not say:

- “BatchNorm solves covariate shift.”
- “Residual connections prevent all vanishing gradients.”
- “ReLU never saturates.”
- “He initialization preserves variance exactly.”
- “Normalization is always necessary.”
- “Deeper is always better.”

Each method changes the optimization geometry; none is universal.

---

### Slide 68 — Retrieval exercise

Complete:

\[
\text{tanh/sigmoid}
\Rightarrow
\underline{\hspace{2cm}}
\]

\[
\text{ReLU}
\Rightarrow
\underline{\hspace{2cm}}
\]

\[
\operatorname{Var}(W)=\frac1{n_{\text{in}}}
\Rightarrow
\underline{\hspace{2cm}}
\]

\[
\operatorname{Var}(W)=\frac2{n_{\text{in}}}
\Rightarrow
\underline{\hspace{2cm}}
\]

\[
x_{\ell+1}=x_\ell+F(x_\ell)
\Rightarrow
\underline{\hspace{2cm}}
\]

Answers:

- saturation;
- zero gradient on negative side;
- Xavier-like forward scaling;
- He initialization;
- residual connection.

---

### Slide 69 — Final mental model

\[
\boxed{
\text{Good initialization places the network in a trainable regime.}
}
\]

\[
\boxed{
\text{Good activations preserve useful nonlinear gradients.}
}
\]

\[
\boxed{
\text{Normalization stabilizes representation scale and optimization.}
}
\]

\[
\boxed{
\text{Residual connections shorten effective learning paths.}
}
\]

---

### Slide 70 — Next lecture

Now the network can be optimized.

Next question:

> How do we prevent it from fitting the training data too well?

Topics:

- generalization;
- train/validation/test splits;
- weight decay;
- dropout;
- early stopping;
- data augmentation;
- label smoothing.

---

# Timing plan

| Section | Slides | Time |
|---|---:|---:|
| Why deep networks fail | 1–7 | 7 min |
| Activations | 8–18 | 12 min |
| Initialization derivation | 19–32 | 19 min |
| Normalization | 33–49 | 19 min |
| Residual connections | 50–60 | 15 min |
| Integration and diagnostics | 61–70 | 8 min |
| **Total** | **70** | **80 min** |

Seventy slides is acceptable here because many are visual/reveal slides. For a smaller deck, merge:

- 9–11;
- 20–22;
- 26–28;
- 37–38;
- 44–46;
- 53–55;
- 64–65.

That yields approximately **57 physical slides**.

# Essential worked examples

## 1. Initialization derivation

\[
z=\sum_{i=1}^n w_ix_i
\]

derive:

\[
\operatorname{Var}(z)
=
n\operatorname{Var}(w)\operatorname{Var}(x).
\]

Then numerically compare:

\[
\sigma_w=0.01,\quad
\sigma_w=1,\quad
\text{Xavier},\quad
\text{He}.
\]

## 2. BatchNorm by hand

Use:

\[
x=[1,2,3,4],
\]

compute:

\[
\mu,\quad\sigma^2,\quad\hat x,\quad
\gamma\hat x+\beta.
\]

## 3. Residual gradient

For:

\[
y=x+wx^2,
\]

derive:

\[
\frac{dy}{dx}=1+2wx.
\]

Explicitly identify:

- identity gradient \(1\);
- residual gradient \(2wx\).

# Interactives

1. **`activation-lab.html`**  
   Function, derivative, saturation/dead-unit percentage and repeated-depth behaviour.

2. **`initialization-signal-flow.html`**  
   Forward activation statistics and backward gradient statistics across up to 200 layers.

3. **`normalization-axes.html`**  
   BatchNorm, LayerNorm and RMSNorm over an editable tensor.

4. **`batchnorm-train-test.html`**  
   Batch statistics versus running statistics and train/eval mismatch.

5. **`residual-gradient-flow.html`**  
   Plain versus residual networks with layer-wise gradient norms.

6. **`deep-network-ablation.html`**  
   Compare all combinations on the same small classification problem.

# Notebooks

1. `01_activation_and_derivatives.ipynb`  
   Plot functions, derivatives, saturation and dead-unit rates.

2. `02_variance_propagation_derivation.ipynb`  
   Empirically verify:
   \[
   \operatorname{Var}(Wx)
   \approx
   n_{\text{in}}\operatorname{Var}(W)\operatorname{Var}(x).
   \]

3. `03_xavier_vs_he.ipynb`  
   Deep random MLP; activation and gradient distributions.

4. `04_normalization_from_scratch.ipynb`  
   Implement BatchNorm, LayerNorm and RMSNorm in NumPy.

5. `05_batchnorm_train_eval.ipynb`  
   Demonstrate running-statistics mistakes.

6. `06_plain_vs_residual.ipynb`  
   Compare gradient flow and optimization as depth increases.

7. `07_deep_mlp_ablation.ipynb`  
   Train six configurations and create the lecture’s summary plots.
