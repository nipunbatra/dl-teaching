---
title: "Lecture 3A: Calculus Toolkit for Deep Learning"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 3A: Calculus Toolkit for Deep Learning

**Title:** *Derivatives, Gradients, Jacobians and Hessians*  
**Subtitle:** *The geometry behind optimization and backpropagation*

Target: **75–80 min, ~45 slides**

---

## Part I — Why this lecture?

### Slide 1 — Title

\[
\textbf{Derivatives, Gradients, Jacobians and Hessians}
\]

> The geometry behind optimization and backpropagation

---

### Slide 2 — Why do we need this?

In DL, we repeatedly compute:

\[
\theta_{t+1}
=
\theta_t-\eta\
abla_\theta L(\theta_t)
\]

So we need to understand:

\[
\frac{d}{dx},\quad
\frac{\partial}{\partial x_i},\quad
\
abla f,\quad
J,\quad
H.
\]

---

### Slide 3 — Big picture

| Object | Function type | Tells us |
|---|---|---|
| derivative | \(\mathbb R\to\mathbb R\) | slope |
| partial derivative | \(\mathbb R^d\to\mathbb R\) | sensitivity to one coordinate |
| gradient | \(\mathbb R^d\to\mathbb R\) | steepest direction |
| Jacobian | \(\mathbb R^n\to\mathbb R^m\) | local linear map |
| Hessian | \(\mathbb R^d\to\mathbb R\) | curvature |

---

# Part II — Scalar derivative

### Slide 4 — Derivative as slope

For:

\[
f:\mathbb R\to\mathbb R
\]

\[
f'(x)=\frac{df}{dx}
=
\lim_{\epsilon\to0}
\frac{f(x+\epsilon)-f(x)}{\epsilon}.
\]

Visual: curve + tangent line.

---

### Slide 5 — Local linear approximation

Near \(x\):

\[
f(x+\Delta x)
\approx
f(x)+f'(x)\Delta x.
\]

Message:

\[
\boxed{\text{derivative = best local linear approximation}}
\]

---

### Slide 6 — Example

\[
f(x)=x^2
\]

\[
f'(x)=2x.
\]

At \(x=3\):

\[
f'(3)=6.
\]

Meaning:

\[
f(3+\Delta x)\approx 9+6\Delta x.
\]

---

### Slide 7 — Interactive 1: tangent line

Demo: `derivative-tangent.html`

Controls:

- function: \(x^2,\sin x,e^x,\log(1+e^x)\)
- point \(x\)
- \(\Delta x\)

Show:

- function curve;
- tangent;
- secant;
- approximation error.

---

# Part III — Partial derivatives and surfaces

### Slide 8 — From curves to surfaces

For:

\[
f:\mathbb R^2\to\mathbb R
\]

example:

\[
f(x,y)=x^2+y^2.
\]

Input is a point \((x,y)\).  
Output is height \(z=f(x,y)\).

Visual: 3D bowl.

---

### Slide 9 — Partial derivative

\[
\frac{\partial f}{\partial x}
\]

means:

> change \(x\), hold \(y\) fixed.

\[
\frac{\partial f}{\partial y}
\]

means:

> change \(y\), hold \(x\) fixed.

For:

\[
f(x,y)=x^2+y^2
\]

\[
\frac{\partial f}{\partial x}=2x,
\qquad
\frac{\partial f}{\partial y}=2y.
\]

---

### Slide 10 — Surface slice view

Show surface \(f(x,y)\).  
Then show two slices:

- fix \(y=y_0\), vary \(x\);
- fix \(x=x_0\), vary \(y\).

Partial derivatives are slopes of these slices.

---

### Slide 11 — Example with interaction

\[
f(x,y)=x^2+3y^2.
\]

At \((x,y)=(1,2)\):

\[
\frac{\partial f}{\partial x}=2
\]

\[
\frac{\partial f}{\partial y}=12.
\]

So the function changes faster along \(y\).

---

### Slide 12 — Interactive 2: surface slices

Demo: `surface-partials.html`

Controls:

- point \((x,y)\)
- function:
  \[
  x^2+y^2,\quad x^2+3y^2,\quad \sin x\cos y
  \]
- show \(x\)-slice;
- show \(y\)-slice.

Display:

- 3D surface;
- two 2D slice plots;
- partial derivative values.

---

# Part IV — Gradient

### Slide 13 — Gradient definition

For:

\[
f:\mathbb R^d\to\mathbb R
\]

\[
\
abla f(x)
=
\begin{bmatrix}
\frac{\partial f}{\partial x_1}\\
\frac{\partial f}{\partial x_2}\\
\vdots\\
\frac{\partial f}{\partial x_d}
\end{bmatrix}.
\]

For \(f(x,y)=x^2+y^2\):

\[
\
abla f(x,y)=
\begin{bmatrix}
2x\\
2y
\end{bmatrix}.
\]

---

### Slide 14 — Gradient as local linear model

\[
f(x+\Delta x)
\approx
f(x)+\
abla f(x)^\top \Delta x.
\]

This is the multidimensional version of:

\[
f(x+\Delta x)\approx f(x)+f'(x)\Delta x.
\]

---

### Slide 15 — Directional derivative

If \(u\) is a unit direction:

\[
D_uf(x)=\
abla f(x)^\top u.
\]

So the change in direction \(u\) is the projection of the gradient onto \(u\).

---

### Slide 16 — Steepest ascent direction

By Cauchy–Schwarz:

\[
\
abla f(x)^\top u
\le
\|\
abla f(x)\|\|u\|.
\]

For \(\|u\|=1\), maximum occurs at:

\[
u=\frac{\
abla f(x)}{\|\
abla f(x)\|}.
\]

Thus:

\[
\boxed{\
abla f=\text{direction of steepest ascent}}
\]

and:

\[
-\
abla f=\text{direction of steepest descent}.
\]

---

### Slide 17 — Contour plots

A contour is:

\[
f(x,y)=c.
\]

Examples:

\[
x^2+y^2=c
\]

gives circles.

Visual:

- surface plot;
- corresponding contour plot.

---

### Slide 18 — Gradient and contours

The gradient is perpendicular to contours.

Reason:

Along a contour, \(f\) does not change.

If \(v\) is tangent to contour:

\[
\
abla f^\top v=0.
\]

So \(\
abla f\) is normal to contour.

Visual: contour lines + gradient arrows.

---

### Slide 19 — Gradient descent on contours

Update:

\[
x_{t+1}=x_t-\eta\
abla f(x_t).
\]

Visual:

- contour plot;
- trajectory moving downhill;
- effect of small vs large \(\eta\).

---

### Slide 20 — Interactive 3: gradient field

Demo: `gradient-contours.html`

Controls:

- function:
  \[
  x^2+y^2,\quad x^2+10y^2,\quad \sin x+\cos y
  \]
- learning rate \(\eta\)
- start point
- number of GD steps

Show:

- surface plot;
- contour plot;
- gradient arrows;
- optimization trajectory.

---

# Part V — Hessian and curvature

### Slide 21 — Second derivative in 1D

For:

\[
f:\mathbb R\to\mathbb R
\]

\[
f''(x)=\frac{d^2f}{dx^2}.
\]

Meaning:

> how fast the slope is changing.

Examples:

\[
f(x)=x^2,\quad f''(x)=2
\]

\[
f(x)=-x^2,\quad f''(x)=-2.
\]

---

### Slide 22 — Convex, concave, flat

Use 1D plots:

- \(f''(x)>0\): bowl;
- \(f''(x)<0\): hill;
- \(f''(x)=0\): locally linear/flat.

---

### Slide 23 — Hessian definition

For:

\[
f:\mathbb R^d\to\mathbb R
\]

\[
H
=
\
abla^2 f
\]

where:

\[
H_{ij}
=
\frac{\partial^2 f}{\partial x_i\partial x_j}.
\]

For two variables:

\[
H=
\begin{bmatrix}
\frac{\partial^2 f}{\partial x^2}
&
\frac{\partial^2 f}{\partial x\partial y}
\\
\frac{\partial^2 f}{\partial y\partial x}
&
\frac{\partial^2 f}{\partial y^2}
\end{bmatrix}.
\]

---

### Slide 24 — Hessian example

\[
f(x,y)=x^2+3y^2.
\]

\[
\
abla f=
\begin{bmatrix}
2x\\
6y
\end{bmatrix}.
\]

\[
H=
\begin{bmatrix}
2&0\\
0&6
\end{bmatrix}.
\]

Curvature is larger in \(y\)-direction.

---

### Slide 25 — Second-order local approximation

\[
f(x+\Delta x)
\approx
f(x)
+
\
abla f(x)^\top\Delta x
+
\frac12\Delta x^\top H(x)\Delta x.
\]

Interpretation:

- gradient: tilt;
- Hessian: curvature.

---

### Slide 26 — Hessian geometry

For quadratic:

\[
f(x)=\frac12x^\top Ax.
\]

\[
\
abla f=Ax
\]

\[
H=A.
\]

Eigenvectors of \(H\): curvature directions.  
Eigenvalues of \(H\): curvature magnitudes.

Visual: elliptical contours.

---

### Slide 27 — Positive definite, negative definite, saddle

At a critical point:

\[
\
abla f=0.
\]

Use Hessian eigenvalues:

| Hessian eigenvalues | Local shape |
|---|---|
| all positive | local minimum |
| all negative | local maximum |
| mixed signs | saddle |
| zeros | inconclusive/flat |

---

### Slide 28 — Saddle point example

\[
f(x,y)=x^2-y^2.
\]

\[
\
abla f=
\begin{bmatrix}
2x\\
-2y
\end{bmatrix}.
\]

\[
H=
\begin{bmatrix}
2&0\\
0&-2
\end{bmatrix}.
\]

At \((0,0)\):

\[
\
abla f=0
\]

but it is not a minimum.

Visual: saddle surface and contours.

---

### Slide 29 — Why Hessian matters in optimization

If contours are circular:

\[
f(x,y)=x^2+y^2
\]

gradient descent behaves nicely.

If contours are elongated:

\[
f(x,y)=x^2+100y^2
\]

gradient descent zigzags.

This is related to condition number / ill-conditioning; high curvature in one direction and low curvature in another can slow first-order methods.

---

### Slide 30 — Interactive 4: curvature and GD

Demo: `hessian-curvature.html`

Controls:

- quadratic:
  \[
  ax^2+by^2+cxy
  \]
- learning rate;
- start point.

Show:

- contour plot;
- surface plot;
- Hessian;
- eigenvectors;
- GD trajectory.

Ask:

> Why does the trajectory zigzag?

---

# Part VI — Jacobian

### Slide 31 — Why gradient is not enough

Gradient is for scalar output:

\[
f:\mathbb R^d\to\mathbb R.
\]

But neural network layers often map vectors to vectors:

\[
f:\mathbb R^n\to\mathbb R^m.
\]

Example:

\[
z=Wx+b.
\]

Need Jacobian.

---

### Slide 32 — Jacobian definition

For:

\[
f:\mathbb R^n\to\mathbb R^m
\]

\[
J
=
\frac{\partial f}{\partial x}
\in\mathbb R^{m\times n}
\]

where:

\[
J_{ij}
=
\frac{\partial f_i}{\partial x_j}.
\]

---

### Slide 33 — Jacobian as local linear map

\[
f(x+\Delta x)
\approx
f(x)+J(x)\Delta x.
\]

This generalizes:

\[
f(x+\Delta x)\approx f(x)+f'(x)\Delta x.
\]

and:

\[
f(x+\Delta x)\approx f(x)+\
abla f^\top\Delta x.
\]

---

### Slide 34 — Jacobian example: linear layer

\[
f(x)=Wx+b.
\]

Then:

\[
J=W.
\]

Because:

\[
\frac{\partial z_i}{\partial x_j}=W_{ij}.
\]

So linear layers are literally local linear maps.

---

### Slide 35 — Jacobian example: elementwise activation

\[
h=\phi(z).
\]

\[
h_i=\phi(z_i).
\]

Jacobian:

\[
J=
\operatorname{diag}
\left(
\phi'(z_1),\ldots,\phi'(z_m)
\right).
\]

For ReLU:

\[
J_{ii}\in\{0,1\}.
\]

---

### Slide 36 — Chain rule with Jacobians

If:

\[
x\rightarrow u\rightarrow v
\]

where:

\[
u=f(x),
\qquad
v=g(u),
\]

then:

\[
\frac{\partial v}{\partial x}
=
\frac{\partial v}{\partial u}
\frac{\partial u}{\partial x}.
\]

So:

\[
J_{v,x}=J_{v,u}J_{u,x}.
\]

---

### Slide 37 — Why we usually avoid full Jacobians

For a layer:

\[
x\in\mathbb R^{4096},
\qquad
h\in\mathbb R^{4096}.
\]

Jacobian size:

\[
4096\times4096
\approx 1.68\times10^7
\]

entries for one layer.

For tensors/images, explicit Jacobians become huge.

D2L notes that while Jacobians are natural for vector-to-vector derivatives, DL frameworks usually compute vector-Jacobian products instead of materializing full Jacobians.

---

### Slide 38 — Vector-Jacobian product

Suppose:

\[
y=f(x),
\qquad
L=L(y).
\]

Then:

\[
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}.
\]

This is:

\[
\bar x
=
\bar y J.
\]

or column convention:

\[
\bar x
=
J^\top \bar y.
\]

This is what backprop repeatedly computes.

---

### Slide 39 — Backprop as VJP chain

For:

\[
x\rightarrow h_1\rightarrow h_2\rightarrow L
\]

\[
\
abla_x L
=
J_{h_1,x}^\top
J_{h_2,h_1}^\top
\
abla_{h_2}L.
\]

Do not form the full matrices.

Compute from right to left:

\[
\bar h_2\rightarrow\bar h_1\rightarrow\bar x.
\]

CS231n similarly frames backprop as modular chain-rule computation on computation graphs, with local gradients passed backward.

---

### Slide 40 — Interactive 5: Jacobian as local warp

Demo: `jacobian-local-map.html`

Use a 2D vector function:

\[
f(x,y)=
\begin{bmatrix}
x+0.5\sin y\\
y+0.5\sin x
\end{bmatrix}.
\]

Show:

- grid in input space;
- warped output grid;
- local Jacobian at selected point;
- small circle transformed into ellipse.

Message:

\[
J = \text{local linear approximation of a vector map}.
\]

---

# Part VII — Where each object appears in DL

### Slide 41 — Derivative

Used for scalar activations:

\[
\sigma'(z),\quad
\tanh'(z),\quad
\operatorname{ReLU}'(z),\quad
\operatorname{GELU}'(z).
\]

Example:

\[
\frac{d}{dz}\sigma(z)=\sigma(z)(1-\sigma(z)).
\]

---

### Slide 42 — Gradient

Used for parameter updates:

\[
\
abla_\theta L.
\]

Gradient descent:

\[
\theta\leftarrow\theta-\eta\
abla_\theta L.
\]

Also used for:

- saliency maps:
  \[
  \
abla_x f_y(x)
  \]
- adversarial examples:
  \[
  x_{\text{adv}}=x+\epsilon\operatorname{sign}(\
abla_xL)
  \]

---

### Slide 43 — Jacobian

Used in:

- backpropagation;
- sensitivity analysis;
- normalizing flows:
  \[
  \log\left|\det J\right|
  \]
- neural ODEs;
- implicit layers;
- representation analysis.

But usually via:

\[
Jv
\quad\text{or}\quad
J^\top v.
\]

---

### Slide 44 — Hessian

Used in:

- curvature analysis;
- Newton method:
  \[
  x_{t+1}=x_t-H^{-1}\
abla f
  \]
- second-order optimization;
- Laplace approximation;
- loss landscape analysis;
- sharpness/flatness;
- influence functions.

But full Hessian is usually too large.

---

### Slide 45 — Why DL mostly uses first-order methods

For \(P\) parameters:

\[
\
abla_\theta L\in\mathbb R^P.
\]

But Hessian:

\[
H\in\mathbb R^{P\times P}.
\]

If \(P=10^8\):

\[
H\text{ has }10^{16}\text{ entries}.
\]

So practical DL usually uses:

- SGD;
- momentum;
- Adam;
- low-rank/diagonal approximations;
- Hessian-vector products when needed.

---

# Part VIII — Autodiff connection

### Slide 46 — Autodiff computes these objects through programs

Automatic differentiation is neither symbolic differentiation nor finite differences; it applies the chain rule to the executed program. Baydin et al.’s survey describes AD as a general family of techniques for efficiently and accurately evaluating derivatives of numeric programs.

---

### Slide 47 — Scalar loss case

Neural network training:

\[
\theta\in\mathbb R^P
\rightarrow
L(\theta)\in\mathbb R.
\]

Reverse-mode autodiff efficiently computes:

\[
\
abla_\theta L.
\]

This is backpropagation.

---

### Slide 48 — Non-scalar output case

If:

\[
y=f(x)\in\mathbb R^m,
\]

then `backward()` needs to know what scalar quantity to differentiate.

In frameworks, one often supplies a vector \(v\) and computes:

\[
v^\top J.
\]

This is why PyTorch complains when `.backward()` is called on a non-scalar tensor without a gradient argument.

---

### Slide 49 — PyTorch mini-demo

Notebook: `jacobian_vjp_hessian.ipynb`

```python
import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)

def f(x):
    return torch.stack([
        x[0] ** 2 + x[1],
        torch.sin(x[0] * x[1])
    ])

y = f(x)

J = torch.autograd.functional.jacobian(f, x)
print(J)
```

Then VJP:

```python
v = torch.tensor([1.0, 3.0])
y.backward(v)
print(x.grad)  # v^T J
```

---

### Slide 50 — Hessian mini-demo

```python
def g(x):
    return x[0] ** 2 + 3 * x[1] ** 2 + x[0] * x[1]

H = torch.autograd.functional.hessian(g, x)
print(H)
```

Expected:

\[
H=
\begin{bmatrix}
2&1\\
1&6
\end{bmatrix}.
\]

---

# Part IX — Summary

### Slide 51 — One table to remember

| Object | Shape | Main DL use |
|---|---:|---|
| \(f'(x)\) | scalar | activation derivative |
| \(\
abla f\) | \(d\) | parameter update |
| \(J\) | \(m\times n\) | vector map sensitivity |
| \(J^\top v\) | \(n\) | backprop |
| \(H\) | \(d\times d\) | curvature |
| \(Hv\) | \(d\) | second-order methods |

---

### Slide 52 — Final mental model

\[
\boxed{
\text{Derivative} = \text{slope}
}
\]

\[
\boxed{
\text{Gradient} = \text{direction of steepest ascent}
}
\]

\[
\boxed{
\text{Jacobian} = \text{local linear map}
}
\]

\[
\boxed{
\text{Hessian} = \text{curvature}
}
\]

\[
\boxed{
\text{Backprop} = \text{efficient chain of VJPs}
}
\]

---

## Recommended flow

Make this **Lecture 3**, then backprop becomes **Lecture 4**.

| Lecture | Topic |
|---:|---|
| 1 | Probabilistic foundations: losses, likelihoods, priors |
| 2 | Linear models to MLPs |
| **3** | **Calculus toolkit for DL** |
| **4** | **Computation graphs, backpropagation, autodiff** |
| 5 | Optimization: SGD, momentum, Adam |
| 6 | Training deep networks: initialization, activations, normalization |

This will flow better than compressing calculus and backprop into one lecture.

---

## Interactives for this lecture

1. `derivative-tangent.html`  
   Tangent, secant, local linear approximation.

2. `surface-partials.html`  
   Surface with \(x\)-slice and \(y\)-slice.

3. `gradient-contours.html`  
   Contours, gradient vectors, GD trajectory.

4. `hessian-curvature.html`  
   Quadratics, Hessian eigenvectors, curvature and zigzagging.

5. `jacobian-local-map.html`  
   2D grid warp and local Jacobian ellipse.

---

## Notebooks

1. `01_derivative_numerical_vs_autodiff.ipynb`
2. `02_surfaces_contours_gradients.ipynb`
3. `03_gradient_descent_on_quadratics.ipynb`
4. `04_hessian_eigen_geometry.ipynb`
5. `05_jacobian_vjp_hessian_pytorch.ipynb`

Yes: make it a full lecture. It will make your backprop lecture much smoother.
