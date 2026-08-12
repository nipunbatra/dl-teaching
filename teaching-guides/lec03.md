# Lecture 3 · Calculus Toolkit · Teaching Guide
*Instructor companion for `lecture3a/L3a-calculus-toolkit.typ`. Audience: undergraduates who know basic algebra and matrix multiplication; no multivariable-calculus course is assumed.*

## The spine

A derivative is a **local linear model**. The derivative, gradient, Jacobian, and Hessian are the same idea for different input/output shapes; backpropagation is the efficient repeated application of vector-Jacobian products.

By the end, students should be able to look at a function signature and answer three questions: **What derivative object has the right shape? What local change does it predict? Which product does autodiff actually compute?**

## Where it sits

- **Builds on Lecture 1:** losses as scalar objectives and parameters as quantities we optimize.
- **Builds on Lecture 2:** a linear layer `z = Wx + b`, elementwise activations, and vector/matrix shapes.
- **Sets up Lecture 4:** computation graphs, reverse-mode autodiff, and backpropagation as a chain of VJPs.
- **Returns in Lectures 5–7:** gradient descent, conditioning, learning rates, momentum/Adam, trainability, and regularization.

## Prerequisites and room setup

Students need:

- scalar functions and function composition;
- vectors, matrices, transposes, dot products, and matrix multiplication;
- the idea that a matrix maps one vector to another;
- enough Python to read a short NumPy or PyTorch cell.

Before class:

1. Open the deck and the two exact companion notebooks:
   - [`03_gradient_descent_on_quadratics.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03A/03_gradient_descent_on_quadratics.ipynb)
   - [`05_jacobian_vjp_hessian_pytorch.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03A/05_jacobian_vjp_hessian_pytorch.ipynb)
2. Keep a board area for a running **shape ledger**: `R→R`, `R^d→R`, `R^n→R^m`, and `R^d→R` again for second derivatives.
3. Do not begin with notation. Begin with the training update `theta_(t+1) = theta_t - eta nabla L(theta_t)` and ask what information the gradient must provide.

## Honest 80-minute route: 55 + 15 + 10

### 0–55 min · Core conceptual route

- **0–8 · One idea, five objects.** Use “Why a calculus lecture?” and “The big picture.” Establish that every object is a local linear description whose shape follows from the function shape.
- **8–17 · Scalar derivative.** Derivative as slope, then immediately upgrade the definition to `f(x + Δx) ≈ f(x) + f'(x)Δx`. Work the `x²` at `x=3` numeric check. Use the local-linearization checkpoint.
- **17–25 · Partials and gradient.** A partial is a one-dimensional slice; the gradient stacks all coordinate sensitivities. Keep asking for shapes before formulas.
- **25–34 · Gradient geometry.** Work the directional derivative, Cauchy–Schwarz argument, contour orthogonality, and one gradient-descent update. The key sentence is: **the gradient points uphill; its negative is the locally steepest downhill direction.**
- **34–43 · Hessian and conditioning.** Derive the Hessian of `x² + 3y²`, connect its eigenvalues to curvature, classify a saddle, and use elongated contours to explain zigzagging. Do not turn this into a full eigenvalue lecture.
- **43–55 · Jacobian, chain rule, VJP.** Start from `R^n→R^m`, build the 2→2 Jacobian row by row, check matrix shapes in the chain rule, and end with `x_bar = J^T y_bar`. This is the bridge to Lecture 4.

If time slips, omit “Where each object shows up in deep learning” and the optional Hessian/Jacobian application slides. Never cut the local-linear-model idea, shape ledger, worked 2→2 Jacobian, or VJP bridge.

### 55–70 min · Two notebook demonstrations

Run these prediction-first. The notebooks are small enough that students should reason before execution rather than watch a coding performance.

#### Notebook A · Quadratic gradient descent (7 minutes)

1. Show the two objectives before running: `x² + y²` and `x² + 10y²`.
2. Ask students to predict which path is straight and which changes sign along one coordinate.
3. For `gamma=10, eta=0.09`, calculate the `y` multiplier on the board: `1 - 2 eta gamma = -0.8`. The negative multiplier predicts the zigzag.
4. Run both cells. Connect the trajectory to Hessian eigenvalues `(2, 2gamma)` and condition number `gamma`.
5. Change only `eta` or `gamma`; ask for a prediction before rerunning.

#### Notebook B · Jacobian, VJP, Hessian (8 minutes)

1. Before execution, derive the first Jacobian row for `f(x) = [x0² + x1, sin(x0 x1)]` at `(1,2)`.
2. Ask for the Jacobian shape and the shapes of `v`, `J^T v`, and `x.grad`.
3. Run the Jacobian cell and reconcile the second row with `cos(2)`.
4. Predict whether `backward(v)` forms the full Jacobian. Run the VJP cell and verify `x.grad == v @ J` (the row-vector form of `J^T v`).
5. Derive `H=[[2,1],[1,6]]` for the scalar `g`, then use PyTorch only as a check.

### 70–80 min · Checks and exit ticket

- **70–74 · Shape check.** Give `f: R^3→R^5` and scalar `L(f(x))`. Students write `J_f in R^(5×3)`, `nabla_f L in R^5`, and `nabla_x L = J_f^T nabla_f L in R^3`.
- **74–77 · Geometry check.** Ask why `nabla f` is perpendicular to a contour and why `nabla f=0` does not guarantee a minimum.
- **77–80 · Exit ticket.** For each of `f: R→R`, `f: R^d→R`, and `f: R^n→R^m`, name the derivative object, its shape, and one DL use. Final prompt: “Why does backprop use products instead of full Jacobians?”

## Teach the deck like this

Keep returning to one table:

| Function | Derivative object | Shape | Local model |
|---|---|---:|---|
| `R→R` | derivative | scalar | `f(x+Δx) ≈ f(x)+f'(x)Δx` |
| `R^d→R` | gradient | `d` | `f(x+Δx) ≈ f(x)+nabla f^T Δx` |
| `R^n→R^m` | Jacobian | `m×n` | `f(x+Δx) ≈ f(x)+JΔx` |
| `R^d→R` | Hessian | `d×d` | adds `1/2 Δx^T H Δx` |

The table should feel inevitable, not memorized. Ask “input shape? output shape?” before revealing every derivative shape. On the Jacobian slides, read rows as output coordinates and columns as input coordinates. On the VJP slide, narrate from the scalar loss backward: the upstream vector arrives at a layer and multiplication by `J^T` sends sensitivity to that layer’s inputs.

## Board derivations worth doing

### 1. Local linearization

For `f(x)=x²` at `x=3`, use `Δx=0.1`:

- linear prediction: `9 + 6(0.1) = 9.6`;
- true value: `3.1² = 9.61`;
- error: `0.01`.

Then use `Δx=1` to show why “local” matters: prediction `15`, truth `16`.

### 2. Gradient and directional derivative

For `f(x,y)=x²+y²` at `(1,2)`, `nabla f=(2,4)`. With unit `u=(3/5,4/5)`, compute `nabla f^T u=4.4`, then compare with `||nabla f||=sqrt(20)≈4.47`. This makes “steepest” quantitative.

### 3. Jacobian and VJP

For `f(x1,x2)=[x1²+x2, sin(x1x2)]`, derive

`J = [[2x1, 1], [x2 cos(x1x2), x1 cos(x1x2)]]`.

At `(1,2)`, use `v=(1,3)` and compute `J^T v`. This exact example is reproduced in Notebook B, so the code verifies the board rather than introducing new arithmetic.

## Checks for understanding

Use these before moving on:

1. **Local model:** Why is a derivative more than the slope of a tangent line?
2. **Gradient:** If `u` is tangent to a contour, what is `nabla f^T u`?
3. **Stationary point:** Give one example with `nabla f=0` that is not a minimum.
4. **Hessian:** For `H=diag(2,20)`, which coordinate constrains a stable learning rate more strongly?
5. **Jacobian:** If `f: R^4→R^7`, what is the shape of `J`?
6. **Chain rule:** If `x∈R^2`, `h∈R^3`, and `y∈R^4`, which multiplication order produces `J_(y,x)`?
7. **VJP:** Why does a scalar training loss make reverse mode attractive when there are millions of parameters?

Expected short answers: local change prediction; zero; e.g. `x²-y²` at the origin; the second coordinate; `7×4`; `J_(y,h) J_(h,x)`; one reverse sweep returns the full parameter gradient without materializing layer Jacobians.

## Where students stumble

- **Derivative, gradient, and Jacobian are treated as unrelated formulas.** Return to the local-linear-model table and function shapes.
- **Jacobian orientation is transposed.** Say “rows are outputs, columns are inputs,” then check `JΔx` has output shape.
- **Gradient direction is confused with the direction to the global minimum.** It is only the locally steepest ascent direction. Elongated quadratic contours make the distinction visible.
- **`nabla f=0` is called a minimum.** Use `x²-y²` and mixed Hessian eigenvalue signs.
- **`v^T J` versus `J^T v` looks contradictory.** They are row- and column-vector notations for the same VJP; PyTorch stores the result with the input’s shape.
- **Autodiff is described as symbolic or numerical differentiation.** It applies exact local derivative rules to the executed computation graph.

## If a student asks

- **“Why does backprop transpose the Jacobian?”** The forward perturbation maps as `Δy=JΔx`. Sensitivities are covectors: preserving the scalar change `y_bar^T Δy = x_bar^T Δx` gives `x_bar=J^T y_bar`.
- **“Does PyTorch ever form a Jacobian?”** It can when explicitly asked for one in a small diagnostic, as the notebook does. Ordinary scalar-loss `.backward()` computes VJPs and avoids the full matrix.
- **“Why not use the Hessian for every update?”** With `P` parameters, the gradient has `P` entries but the dense Hessian has `P²`; forming, storing, and inverting it is usually prohibitive.
- **“Why does bad conditioning cause zigzagging?”** One shared step size must be small enough for the high-curvature direction, so progress along the low-curvature direction is slow; near the stability limit the steep coordinate alternates signs.

## Instructor verification checklist

- The public Lecture 3 row points to `slides-pdf/L3A.pdf`, not the separate backprop deck used for public Lecture 4.
- Both public Colab links use the exact `notebooks/L03A/` filenames listed above.
- The deck’s local arithmetic matches the notebook outputs: the Jacobian second row at `(1,2)` is `[2 cos(2), cos(2)]≈[-0.8323,-0.4161]`; with `v=(1,3)`, the VJP is approximately `[-0.4969,-0.2484]`; the Hessian is `[[2,1],[1,6]]`.
- The notebooks execute top to bottom with no errors.
- The handout and progressive PDF render without clipped text, broken links, or unreadable equations.

## Closing line

“One idea changes shape: slope for a scalar, direction for a scalar field, a local map for a vector function, curvature for a scalar field — and backprop moves those sensitivities backward without building the maps.”
