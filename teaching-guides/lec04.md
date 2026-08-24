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
| Symbolic differentiation | **formula-to-formula calculus** | Keeping the variables symbolic—by hand or in a computer-algebra system—produces a derivative formula before any values are substituted. |
| Central finite differences | **numerical diagnostic** | A few coordinates approximately match the derivative value from symbolic differentiation and autodiff; this is a check, not a training method. |
| PyTorch outputs | **executed software evidence** | Autograd returns the same derivatives on the same graph and accumulates repeated backward calls. |
| Two-output dense layer | **constructed teaching example** | `z=Wx+b` is two familiar neurons stacked; exact VJPs expose `dx`, `dW`, and `db`. |
| Three-example dense batch | **constructed and exactly computed** | Shared-parameter contributions add, the declared mean reduction divides by three, and scaled microbatches equal one full-batch backward pass. |
| Computation-graph drawings | **explanatory diagrams** | Dependency and reverse-flow structure; they are not empirical measurements. |

This sequence matters pedagogically: students first predict a result, calculate it by hand, and only then let PyTorch verify it. Avoid presenting a printed code output as proof when the notebook can execute the check and assert the result.

## 80-minute route

### 0–10 min · From a loss to an executed graph

- Reconnect to the Lecture 2 model: `x → f_theta(x) → y_hat → L`.
- Establish the diagram grammar: circles store values, boxes transform them, arrows show dependencies.
- Compare symbolic differentiation, central differences, and reverse-mode autodiff by naming what each consumes and returns. Symbolic differentiation transforms one formula into another formula. Finite differences probe nearby function values to approximate one derivative value. Reverse-mode AD records the operations that executed and propagates derivative values backward through their local rules.

Speaker line: “Pen and paper is a medium, not a differentiation method. I can do symbolic differentiation or a small reverse sweep on paper; a computer can do either one too.”

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

Use the same example in the slides and focused scalar notebook:

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

#### Focused scalar-engine Colab · PyTorch first, then from scratch

Use [`04_scalar_autograd_pytorch_to_scratch.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03/04_scalar_autograd_pytorch_to_scratch.ipynb) as the in-class scalar demonstration. In the live eight-minute core, establish the PyTorch answer, then expose the same backward pass in the tiny scratch engine:

1. Create `w`, `x`, `b`, and `y` with literal `torch.tensor(..., requires_grad=True)` lines.
2. Run the four-line forward graph, call `L.backward()`, and read the `.grad` values directly.
3. Build a minimal `Value` record containing `data`, an accumulated `grad` buffer, and explicit `ParentLink(value, local_grad)` entries.
4. Inspect what the two parent links of `m = w*x` saved during the forward pass.
5. Establish the readiness rule before naming it: append a value only after its dependencies, call the result a dependency-first or topological order, then reverse it for backward. Step through the [hosted topological-order interactive](https://nipunbatra.github.io/interactive-articles/autograd-topological-order/) embedded in the notebook; it receives the validated trace of the actual `sL` graph. Keep the highlighted code line, call stack, `seen`, active `ParentLink`, and growing list visible together. End with the exact schedules `w,x,m,b,a,y,e,L` and `L,e,y,a,b,m,x,w`.
6. Seed `L.grad = 1`, then read all seven reverse-edge cards using the lecture convention: teal upstream from `node.grad`, blue local derivative from the parent link, and the orange edge contribution added to `parent.grad`.
7. Compare the Graphviz view before and after `backward()`, inspect the complete cards for all eight `Value`s and seven `ParentLink`s, then check every named value and gradient against PyTorch.

If time permits—or as a follow-up—run Part 3. It builds the neuron `z = w*x + b`, then compares one fused `sigmoid` parent link with the four atomic links in `1 / (1 + exp(-z))`. Show all 8 fused and 11 atomic reverse edges; highlight that the activation itself takes 1 update versus 4 while both return `z.grad = -0.25`.

Keep branch accumulation, finite differences, and the optimizer update in the complete lecture notebook below. The focused notebook stops after this one controlled fusion comparison.

#### Complete lecture Colab · follow-along and after class

1. Predict all stored forward values.
2. Fill the reverse ledger before running the backward cell.
3. Compare with symbolic differentiation: derive the general formula first, then substitute the numbers.
4. Run the central-difference check for `w`, `b`, `x`, and `y`.
5. Execute the `eta=0.01` update and assert that the new loss is smaller.
6. Predict `.grad` for every tracked leaf before `backward()`.
7. Retain intermediate gradients only for inspection; explain why ordinary training does not need them.
8. Call backward on fresh graphs without clearing and watch the leaf buffer accumulate.
9. Verify that the branch `x² + 3x` returns `8 + 3 = 11` at the shared `x`.
10. Run the vector-neuron checkpoint before moving to the dense layer.

Full-lecture companion: [`00_backprop_autograd_complete.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L03/00_backprop_autograd_complete.ipynb)

### 62–70 min · One neuron → a dense-layer VJP

Slow the transition down and keep the same visual grammar:

1. Treat row 1 of `W` as one familiar neuron: one dot product plus one bias.
2. Add row 2 and stack the two stored outputs into `z=Wx+b`.
3. Explicitly state that the later loss sends `g_z=(4,-2)` back to this layer.
4. Reuse the scalar affine rule for output row `i`: `x_j` receives `g_i W_ij`, `W_ij` receives `g_i x_j`, and `b_i` receives `g_i`.
5. Add the two returns to the shared input, then recognize `(g_x)_j=Σ_i g_iW_ij` as `g_x=W^Tg_z`.
6. Stack the two separate parameter rows to obtain `g_W=g_zx^T`; each bias has local derivative one, so `g_b=g_z`.
7. Only after the coordinate derivation, verify operand shapes with generic `m,d`. Shape matching checks the transpose and multiplication order; it is not the derivation.

Use `x=(2,-1)`, `W=[[1,3],[-2,1]]`, and `b=(0,1)` throughout the single-example calculation. It gives `z=(-1,-4)`; with arriving `g_z=(4,-2)`, the exact returns are `g_x=(8,10)`, `g_W=[[8,-4],[-4,2]]`, and `g_b=(4,-2)`.

### 70–78 min · From one example to a batch

Keep the operations familiar and introduce only one new idea at a time:

1. State the convention switch explicitly: an individual `x^(n)` is a `d×1` column, while a framework stacks example transposes as rows `X∈R^(B×d)`.
2. Show `Z=XW^T+1_Bb^T`, or `X @ W.T + b` in PyTorch. Batching adds an example axis; every row still reuses the same `W,b`.
3. Walk the three-row ledger in order: `x^(n) → z^(n) → y^(n) → r^(n) → ell_n`. Point out that `r^(1)=(4,-2)` is exactly the arriving vector from the preceding single-example calculation.
4. Derive the missing bridge: for `ell_n=1/2||z^(n)-y^(n)||²`, `∂ell_n/∂z^(n)=r^(n)`. Since `L=(ell_1+ell_2+ell_3)/3`, the batch arrival is `G_Z=R/3`.
5. Reuse the single-example dense rules on example 1 before exposing the other rows. Its unscaled returns are `G_W^(1)=[[8,-4],[-4,2]]`, `g_b^(1)=(4,-2)`, and `g_x^(1)=(8,10)`; the mean-loss contribution is each divided by three.
6. Add the three paths into the shared parameters, then apply the mean: `g_W=[[4,-2],[-2,4]]` and `g_b=(1,1)`. Keep the two operations verbally distinct: **paths add; the reduction scales**.
7. Stack only at the end: `g_W=G_Z^T X`, `g_b=G_Z^T1_B`, and `g_X=G_ZW`. The rows of `g_X` remain separate because the three inputs are distinct graph values.
8. Contrast sum and mean reductions. The local dense rule is unchanged; only the upstream scale differs by `B=3`.

#### Continue the same Colab · checkpoints 11–16

1. Predict the first output from row 1 before revealing the dot product.
2. Stack the second row and verify `z=Wx+b`.
3. Derive the coordinate rules, expose the two row contributions to `g_x`, then let PyTorch reproduce every entry.
4. Audit the `X,Z,Y,R,ell` ledger and verify `Z.grad=R/3` for the declared mean loss.
5. Inspect all three per-example `r^(n)(x^(n))^T` contributions before averaging them.
6. Verify `X.grad=RW/3`; unlike shared `W,b`, distinct input rows do not add into one another.
7. Accumulate three correctly scaled microbatches and assert exact agreement with the full batch.

The complete lecture notebook then verifies one optimizer step and the final tiny MLP, so the vector-to-batch sequence stays in one document.

If time is short, compute the first example and the final mean on the board, then let checkpoints 12–14 expose the other two contributions. Treat microbatch equivalence as an extension. Never skip the residual derivative, the row/column convention, or the distinction between a per-example contribution, its sum, and the reduction used by the scalar loss.

### 78–80 min · Exit ticket

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
- **They equate symbolic with pen and paper.** Symbolic describes formula-to-formula manipulation, whether a person or a computer-algebra system performs it. A reverse sweep can also be worked by hand.
- **They think autograd is symbolic algebra.** Autograd does not normally construct or simplify a general derivative formula. It records the operations that executed and evaluates their local derivative rules in reverse at the current values.
- **They expect every intermediate tensor to expose `.grad`.** Leaf gradients are retained by default; intermediate gradients need `retain_grad()` when inspected for teaching or debugging.
- **They forget accumulation between optimizer steps.** `backward()` adds to `.grad`; `zero_grad()` clears the boundary between intended updates.
- **They see a matrix multiply as one opaque rule.** Read each row of `W` as one neuron first; only then stack the outputs.
- **They invent the dense layer's arriving gradient.** Say where it comes from: the later loss sends the vector `g_z` back, just as the square example was given an upstream scalar.
- **They treat shape matching as a proof.** Derive the coordinate returns first. Shapes verify orientation and expose a bad transpose, but several wrong formulas can share the same numeric shape when `m=d=2`.
- **They mix single-example columns with batch rows.** Keep `x^(n)` as `d×1`, but state that PyTorch stacks `(x^(n))^T` into `X∈R^(B×d)`; this is why the batch forward is `XW^T+b`.
- **They mix sum and mean reductions.** Shared paths always add. A mean loss scales that sum by the batch size; one-example microbatches must use the same overall scaling to match a full batch.
- **They call a successful finite-difference check a proof.** It is a numerical diagnostic at chosen coordinates and tolerances, not a proof of a whole implementation.

## If a student asks

- **“Why reverse mode?”** A scalar loss has one output but possibly millions of parameters. One reverse sweep returns sensitivities for all of them; forward-mode would be attractive for few inputs and many outputs instead.
- **“Is symbolic differentiation just the analytical method we use on paper?”** It is the analytical, formula-to-formula method often used on paper, but paper is not essential. A computer-algebra system can symbolically derive the same formula. Conversely, we can execute a tiny reverse-mode graph by hand without making it symbolic differentiation.
- **“Does PyTorch build the same graph every time?”** It records the operations executed in that forward pass. Python control flow can therefore produce a different dynamic graph on the next pass.
- **“Why save forward values?”** Many local derivatives need them: multiplication needs the other operand, square needs its input, and an activation may need its pre-activation or output.
- **“Why does the target have a derivative in the board example?”** We track it to audit the complete graph. Ordinary supervised training treats targets as fixed data, so they do not require gradients.
- **“Is backprop the optimizer?”** No. Backprop computes gradients. The optimizer applies an update using those gradients and its own state.
- **“Why not use finite differences everywhere?”** They are approximate and require two additional loss evaluations for every checked coordinate. Use them sparsely to debug symbolic or autodiff derivative values.

## Instructor verification checklist

- The focused scalar notebook executes top to bottom with 11 sequential code cells, zero errors, no warning outputs, three embedded teaching diagrams, one self-contained interactive ordering trace, four Graphviz SVGs, complete state cards for all eight square-example `Value`s and seven `ParentLink`s, and three complete seed-to-leaf traces: 7 edges for the square example, 8 for fused sigmoid, and 11 for atomic sigmoid.
- The first trace consistently shows upstream × local = downstream contribution and the parent buffer before/after each update. It reports forward values `6, 7, -3, 9` and gradients `w=-18`, `b=-6`, `x=-12`, `y=6`, then verifies all eight scratch values and gradients against PyTorch.
- The sigmoid comparison verifies `z=0`, `s=0.5`, `L=0.25`, and common gradients `w=-0.5`, `x=-0.125`, `b=z=-0.25`, `y=1`. Four atomic activation locals multiply to the fused local `s(1-s)=0.25`, and both graphs match `torch.sigmoid`.
- The complete notebook retains its 16 lecture checkpoints; central differences agree with all four scalar gradients, and `eta=0.01` lowers the loss from `9` to `5.76` while also demonstrating `.grad` accumulation.
- The complete notebook verifies the branched result `x.grad=8+3=11`, the vector neuron, and the dense result `z=(-1,-4)`, `g_x=(8,10)`, `g_W=[[8,-4],[-4,2]]`, `g_b=(4,-2)`.
- The complete notebook shows why the formulas hold: two row contributions add into `g_x`, while separate weight rows stack into `g_W`; generic operand shapes then verify the result.
- For its three-example mean loss, the complete notebook reports `L=7`, per-example `dW` contributions `[[8,-4],[-4,2]]`, `[[2,-4],[-4,8]]`, and `[[2,2],[2,2]]`, then `W.grad=[[4,-2],[-2,4]]` and `b.grad=(1,1)`.
- The complete notebook verifies `Z.grad=R/3` and `X.grad=[[8/3,10/3],[-10/3,-2/3],[-1/3,4/3]]` for the mean reduction.
- Three losses scaled by `1/3` accumulate to exactly the same `W.grad` and `b.grad` as one full-batch backward call.
- One concrete `lr=0.01` batch update lowers the loss from `7` to `6.5865`; the deterministic tiny MLP has finite, shape-matched gradients and a zero return from its inactive ReLU unit.
- The public Lecture 4 row, both PDFs, the notebook inventory, and the DL26 site distinguish the focused scalar-engine Colab from the complete lecture Colab.

## Closing line

“Forward stores values; backward seeds the loss, multiplies by each local derivative, and adds at branches. That one rule is backprop.”
