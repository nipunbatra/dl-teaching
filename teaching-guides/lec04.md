# Lecture 4 · Computation Graphs & Backpropagation · Teaching Guide
*Instructor companion for the public Lecture 4. Audience: undergraduates who have completed Lectures 1–3.*

## The spine

A forward pass executes a graph and stores values. A backward pass seeds the scalar loss with `1`, sends each arriving sensitivity through a local derivative, and **adds** contributions where paths meet.

The sentence to repeat is:

> **downstream gradient = upstream gradient × local gradient; branches add**

That local rule scales from the hand-worked scalar graph to dense layers and neural networks. Reverse mode is the mathematical direction, backpropagation is the reverse-sweep algorithm, and autograd is the software engine that records and replays the executed operations.

## Where it sits

- **Lecture 1:** the loss is the scalar whose sensitivities we want.
- **Lecture 2:** a neural network is a composition of affine maps and nonlinearities.
- **Lecture 3 · Calculus Toolkit:** the chain rule and vector–Jacobian products provide the mathematical language.
- **Lecture 5 · Optimization for Deep Learning:** begins with the minibatch gradient that backprop returns, then asks how an optimizer should use it.

Do not teach optimizer choice, momentum, or Nesterov here. This lecture is about *computing* gradients; Lecture 5 is about *using* them.

## Learning contract

By the end, students should be able to:

1. decompose a scalar expression into stored values and primitive operations;
2. run the forward pass and one reverse sweep by hand;
3. distinguish an upstream gradient, a local derivative, and the downstream gradient;
4. explain why gradients add at a branch and why implementations accumulate with `+=`;
5. reproduce the four derivatives of `L=(wx+b-y)^2` for the lecture's numeric example;
6. check a derivative with a central finite difference;
7. map graph concepts to `Tensor`, `grad_fn`, `backward()`, and `.grad`;
8. read one row of a dense layer as one neuron, then compute `dx`, `dW`, and `db` from an arriving vector; and
9. explain why examples sharing `W,b` add gradient contributions, how a mean loss scales that sum, and when microbatch accumulation equals full-batch backprop.

## Honest evidence route: construct → calculate → verify

Keep the status of each piece of evidence explicit.

| Evidence | Status | What it supports |
|---|---|---|
| `x=3, w=2, b=1, y=10` | **constructed teaching example** | A graph small enough to audit end to end; it is not a dataset or benchmark. |
| Forward values and hand derivatives | **exactly computed** | The chain rule, local rules, signs, and branch accumulation. |
| Direct symbolic derivatives | **independent algebraic check** | Backprop agrees with differentiating the composed expression directly. |
| Central finite differences | **numerical diagnostic** | A few coordinates agree approximately with the analytic result; this is a check, not a training method. |
| PyTorch outputs | **executed software evidence** | Autograd returns the same derivatives on the same graph and accumulates repeated backward calls. |
| Two-output dense layer | **constructed teaching example** | `z=Wx+b` is two familiar neurons stacked; exact VJPs expose `dx`, `dW`, and `db`. |
| Three-example dense batch | **constructed and exactly computed** | Shared-parameter contributions add, the declared mean reduction divides by three, and scaled microbatches equal one full-batch backward pass. |
| Computation-graph drawings | **explanatory diagrams** | Dependency and reverse-flow structure; they are not empirical measurements. |

This sequence matters pedagogically: students first predict a result, calculate it by hand, and only then let PyTorch verify it. Avoid presenting a printed code output as proof when the notebook can execute the check and assert the result.

## 80-minute route

### 0–10 min · From a loss to an executed graph

- Reconnect to the Lecture 2 model: `x → f_theta(x) → y_hat → L`.
- Establish the diagram grammar: circles store values, boxes transform them, arrows show dependencies.
- Compare symbolic differentiation, central differences, and reverse-mode autodiff. All can estimate the same derivative, but only reverse mode returns every parameter sensitivity in one sweep from a scalar loss.

Board question: “If a model has one million parameters and one scalar loss, why is perturbing every parameter twice a poor training algorithm?”

### 10–24 min · One reusable local rule

- Name upstream, local, and downstream before showing a formula.
- Work the square, addition, multiplication, and ReLU rules.
- At ReLU, say that the ordinary derivative is undefined at exactly zero and software chooses a convention; commonly it returns zero.
- Ask the inactive-ReLU checkpoint before revealing the answer.

Keep the numerator of every derivative as the final loss. `∂L/∂u` says what is being measured and where the sensitivity is stored.

### 24–34 min · Branches accumulate

- Draw a value consumed by two operations.
- Compute each path's chain-rule product separately.
- Add the products and connect the sum to `+=` in an autograd engine.

The key distinction is:

- an addition **operation** copies its arriving gradient to each input;
- a **branch in the graph** adds all contributions that return to the shared value.

### 34–54 min · Central worked graph

Use the same example in the slides and notebooks:

`m=wx`, `a=m+b`, `e=a-y`, `L=e²`, with `x=3`, `w=2`, `b=1`, `y=10`.

Forward ledger:

| Stored value | Calculation | Value |
|---|---|---:|
| `m` | `2×3` | `6` |
| `a` | `6+1` | `7` |
| `e` | `7-10` | `-3` |
| `L` | `(-3)²` | `9` |

Backward ledger:

| Stored value | Derivative of the loss | Why |
|---|---:|---|
| `L` | `1` | reverse seed |
| `e` | `-6` | `1 × 2e` |
| `a` | `-6` | subtraction local derivative `1` |
| `y` | `6` | subtraction local derivative `-1` |
| `m` | `-6` | addition copies |
| `b` | `-6` | addition copies |
| `w` | `-18` | `(-6) × x` |
| `x` | `-12` | `(-6) × w` |

Have students predict the sign of the `w` update. With `eta=0.01`, gradient descent gives `w=2.18`, `b=1.06`, prediction `7.60`, and loss `5.76`, down from `9`. State the limited claim: one sufficiently small step lowers this constructed loss; backprop itself does not choose the learning rate.

### 54–62 min · Hand calculation meets autograd

#### Notebook A · Manual scalar backprop

1. Predict all stored forward values.
2. Fill the reverse ledger before running the backward cell.
3. Compare with direct calculus.
4. Run the central-difference check for `w`, `b`, `x`, and `y`.
5. Execute the `eta=0.01` update and assert that the new loss is smaller.

Companion: [`01_manual_scalar_backprop.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03/01_manual_scalar_backprop.ipynb)

#### Notebook B · PyTorch autograd

1. Predict `.grad` for every tracked leaf before `backward()`.
2. Retain intermediate gradients only for inspection; explain why ordinary training does not need them.
3. Verify the hand-derived derivatives.
4. Call backward on fresh graphs without clearing and watch the leaf buffer accumulate.
5. Clear once, rerun, and connect the reset to `optimizer.zero_grad()`.

Companion: [`02_autograd_scalar.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03/02_autograd_scalar.ipynb)

### 62–76 min · One neuron → dense layer → shared batch

Slow the transition down and keep the same visual grammar:

1. Treat row 1 of `W` as one familiar neuron: one dot product plus one bias.
2. Add row 2 and stack the two stored outputs into `z=Wx+b`.
3. Explicitly state that the later loss sends `g_z=(4,-2)` back to this layer.
4. Return `g_x=W^Tg_z`, `g_W=g_zx^T`, and `g_b=g_z`; read the shapes before the numbers.
5. Keep `W,b` fixed and let three examples use them. Compute each outer-product contribution before adding.
6. Name the scalar reduction: `L=(1/3)Σ_i ½||z_i-y_i||²`. Therefore the shared gradients are the **mean** of the three contributions.
7. Finish by accumulating three losses scaled by `1/3` and matching one full-batch backward call.

Use `x=(2,-1)`, `W=[[1,3],[-2,1]]`, and `b=(0,1)` throughout the single-example calculation. It gives `z=(-1,-4)`; with arriving `g_z=(4,-2)`, the exact returns are `g_x=(8,10)`, `g_W=[[8,-4],[-4,2]]`, and `g_b=(4,-2)`.

#### Notebook C · Dense layer and a batch

1. Predict the first output from row 1 before revealing the dot product.
2. Stack the second row and verify `z=Wx+b`.
3. Calculate the vector backward rules and let PyTorch reproduce every entry.
4. Inspect all three per-example `g_i x_i^T` contributions.
5. Compare their sum with the gradient of the declared mean loss.
6. Accumulate three correctly scaled microbatches and assert exact agreement with the full batch.

Companion: [`03_dense_layer_batch_autograd.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03/03_dense_layer_batch_autograd.ipynb)

If time is short, compute the first example and the final mean on the board, then let Notebook C expose the other two contributions. Never skip the distinction between a per-example contribution, its sum, and the reduction used by the scalar loss.

### 76–80 min · Exit ticket

Ask students to answer without code:

1. Why is `∂L/∂L=1` the correct reverse seed?
2. If a value feeds two downstream operations, what combines at that value?
3. What does `optimizer.step()` do that `loss.backward()` does not?
4. Why is a central finite difference useful for checking but unsuitable for training a million-parameter model?
5. If three examples share `W` and the loss uses a mean, how is `W.grad` related to the three per-example gradients?

Expected answers: identity derivative; the two path contributions add; it changes parameter values using already-computed gradients; it needs two extra forward passes per checked coordinate and is approximate; `W.grad` is the sum of the contributions divided by three.

## Teach it like this

Keep one three-colour verbal grammar throughout:

- **upstream:** sensitivity arriving from the final loss;
- **local:** derivative of this one operation using stored forward values;
- **downstream:** product sent toward the operation's inputs.

At every node, ask the same three questions:

1. What value was stored in the forward pass?
2. What gradient arrives from upstream?
3. What local derivative turns it into each input's contribution?

This rhythm is more durable than memorizing a long symbolic derivative. It also maps directly to what an autograd engine saves and replays.

## Board derivations worth doing

### 1. Why branches add

For `u=x²` and `v=3x`, with `L=u+v`, write both paths:

`dL/dx = (dL/du)(du/dx) + (dL/dv)(dv/dx) = 1·2x + 1·3`.

At `x=2`, the contributions are `4` and `3`, so the accumulated gradient is `7`. Emphasize that overwriting one contribution would be a correctness bug.

### 2. Central difference for one coordinate

For `L(w)=(3w+1-10)²`, use

`g_FD = [L(w+epsilon)-L(w-epsilon)]/(2 epsilon)`.

At `w=2`, the result is approximately `-18`. Use float64 and a small but not vanishing `epsilon`; the companion notebook checks all four scalar inputs.

### 3. One descent step

From `dL/dw=-18` and `dL/db=-6`, calculate the `eta=0.01` update before revealing the new loss. The negative derivatives imply both parameters increase because gradient descent subtracts them.

## Where students stumble

- **They reverse the slogan.** The arriving gradient is multiplied by the derivative of the local output with respect to the local input.
- **They overwrite a branch contribution.** Use `+=`, not `=`; each downstream path contributes one term.
- **They forget the reverse seed.** `dL/dL=1` starts the sweep.
- **They confuse a value with the operation that produced it.** Values store forward numbers and gradients; operation nodes store a local backward rule and any forward values that rule needs.
- **They think autograd is symbolic algebra.** It differentiates the operations that actually executed and applies their local rules in reverse.
- **They expect every intermediate tensor to expose `.grad`.** Leaf gradients are retained by default; intermediate gradients need `retain_grad()` when inspected for teaching or debugging.
- **They forget accumulation between optimizer steps.** `backward()` adds to `.grad`; `zero_grad()` clears the boundary between intended updates.
- **They see a matrix multiply as one opaque rule.** Read each row of `W` as one neuron first; only then stack the outputs.
- **They invent the dense layer's arriving gradient.** Say where it comes from: the later loss sends the vector `g_z` back, just as the square example was given an upstream scalar.
- **They mix sum and mean reductions.** Shared paths always add. A mean loss scales that sum by the batch size; one-example microbatches must use the same overall scaling to match a full batch.
- **They call a successful finite-difference check a proof.** It is a numerical diagnostic at chosen coordinates and tolerances, not a proof of a whole implementation.

## If a student asks

- **“Why reverse mode?”** A scalar loss has one output but possibly millions of parameters. One reverse sweep returns sensitivities for all of them; forward-mode would be attractive for few inputs and many outputs instead.
- **“Does PyTorch build the same graph every time?”** It records the operations executed in that forward pass. Python control flow can therefore produce a different dynamic graph on the next pass.
- **“Why save forward values?”** Many local derivatives need them: multiplication needs the other operand, square needs its input, and an activation may need its pre-activation or output.
- **“Why does the target have a derivative in the board example?”** We track it to audit the complete graph. Ordinary supervised training treats targets as fixed data, so they do not require gradients.
- **“Is backprop the optimizer?”** No. Backprop computes gradients. The optimizer applies an update using those gradients and its own state.
- **“Why not use finite differences everywhere?”** They are approximate and require two additional loss evaluations for every checked coordinate. Use them sparsely to debug analytic or autodiff gradients.

## Instructor verification checklist

- The manual notebook reports forward values `6, 7, -3, 9` and gradients `w=-18`, `b=-6`, `x=-12`, `y=6`.
- Its central differences agree with all four gradients within the declared tolerance.
- With `eta=0.01`, the notebook reports prediction `7.60` and loss `5.76 < 9`.
- The scalar autograd notebook executes top to bottom, reproduces the tracked `w,b` leaf gradients, and demonstrates accumulation on fresh graphs.
- The dense/batch notebook executes top to bottom with sequential counts and zero errors. It reports `z=(-1,-4)`, `g_x=(8,10)`, `g_W=[[8,-4],[-4,2]]`, and `g_b=(4,-2)` for the single example.
- For its three-example mean loss, it reports `L=7`, per-example `dW` contributions `[[8,-4],[-4,2]]`, `[[2,-4],[-4,8]]`, and `[[2,2],[2,2]]`, then `W.grad=[[4,-2],[-2,4]]` and `b.grad=(1,1)`.
- Three losses scaled by `1/3` accumulate to exactly the same `W.grad` and `b.grad` as one full-batch backward call.
- The public Lecture 4 row links the handout, presentation, and all three exact `notebooks/L03/` Colabs.

## Closing line

“Forward stores values; backward seeds the loss, multiplies by each local derivative, and adds at branches. That one rule is backprop.”
