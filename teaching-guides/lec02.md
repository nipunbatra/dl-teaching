# Lecture 2 · From Linear Models to Neural Networks · Teaching Guide

*Instructor companion for the current public Lecture 2 deck.*

## The spine

Lecture 1 chose losses from probability; Lecture 2 expands the prediction rule
$f_\theta(x)$. Start with one affine score, show exactly why affine layers alone
cannot represent XOR, then build and verify one two-unit ReLU MLP before widening
the view to universal approximation, depth, minibatches, and a parameter update.

## What students should leave able to do

- Track the dimensions of a weighted sum, a batch matrix multiplication, and a
  two-layer MLP.
- Separate the affine pre-activation $u$ from the activation $h=\phi(u)$.
- Explain why stacking linear layers still gives a linear map and why XOR defeats
  a single straight decision boundary.
- Evaluate the deck's exact XOR network on all four inputs.
- State the universal-approximation claim without turning existence into a claim
  about learnability or generalization.
- Explain width as feature variety and depth as staged composition.
- Trace one practical forward pass from inputs to class scores, softmax, loss, a
  minibatch, and one numerical parameter update.

## Evidence language for this lecture

Use the deck tags as route markers: `V` is a visual explanation, `D` is a worked
derivation, `Q/A` is a prediction-and-reveal pair, `I` is an interactive, and
`OPT` is optional. They do **not** by themselves imply empirical evidence.

| Material | Honest evidence label | How to say it |
|---|---|---|
| Weighted-sum, batch, neuron, XOR, loss, parameter-count, and update arithmetic | **Computed from displayed values** | “Let us verify the numbers.” |
| XOR inputs and the exact two-unit weights | **Constructed teaching example** | “These weights are chosen to realize XOR; an optimizer did not discover them here.” |
| Linear-regression notebook data | **Synthetic data; computed fit** | “This checks the algorithm against the closed-form solution on seeded synthetic data.” |
| Universal approximation | **Mathematical representation result** | “Approximating weights exist; the theorem does not promise that training finds them.” |
| Node diagrams, decision-region sketches, and practical image pipeline | **Schematic** | “This diagram explains structure; it is not a measured model trace.” |
| Vision/audio/text depth hierarchy graphics | **Conceptual schematic** | “This is one plausible compositional story, not observed activations.” |

Do not call the XOR construction a learned result, the synthetic regression a
real-data experiment, or the hierarchy graphics evidence of what a trained model
actually represents.

## 85-minute route through the current deck

### 0–6 min · Retrieve Lecture 1 and change only the prediction rule

Use “Lecture 1 chose the loss; Lecture 2 expands the prediction rule” as the
handoff. Keep $\hat y=f_\theta(x)$ visible and tell students that the loss and
training loop can remain while the function class becomes richer.

### 6–18 min · Linear starting point

Work through the single weighted sum, every contribution entering the node, the
labelled batch matrix, the matrix multiplication, and the bias-as-ones column.
Close with the regression slide: one affine rule is one flat trend. Make students
say the shapes aloud before revealing them.

### 18–31 min · From a score to a neuron

Separate the two stages: $u=w^Tx+b$, then $h=\phi(u)$. Compare sigmoid/tanh with
ReLU/GELU, use the saturation calculation to give “locally flat” a numerical
meaning, and ask the linear-layers question before revealing that two affine maps
collapse to one. Use the activation interactive only after the algebraic point is
secure.

### 31–42 min · Where one weighted sum fails

Draw the binary straight boundary and let students try to separate XOR. Then
connect the same geometry to multiclass scores: score ties live on flat sets, and
only the top-scoring tie is a visible boundary. The five explicit inputs are a
calculation, not a dataset benchmark.

### 42–57 min · Build one exact XOR MLP

Keep the same network throughout:

$$
h_1=\operatorname{ReLU}(x_1+x_2),\qquad
h_2=\operatorname{ReLU}(x_1+x_2-1),\qquad
z=2h_1-4h_2-1.
$$

Verify all four rows, derive both hidden functions before substituting an input,
then compute the output probability and loss. The hidden-space coordinate slide
is the payoff: the nonlinearity changes the representation so one output node can
separate the cases. Say explicitly that the weights are constructed and the
forward values are computed.

### 57–70 min · Universal approximation via hinges

Frame one precise question, build a ReLU hinge, and assemble the tent function.
The single three-panel **constructed-interpolant** comparison makes the
qualitative point that more hinges reduce the worst gap; `sup` means the largest
gap over the entire displayed region. Stress that these curves use fixed knots
and are not trained results. Finish by reading the theorem in order and dwelling
on the existence warning:
existence is not learnability, data efficiency, or generalization.

### 70–77 min · Width versus depth

Use the node view first: width adds parallel features; depth composes features in
stages. Work the clipped nonlinear ramp so “composition” is an operation, not a
metaphor. The vision, audio, and text hierarchy images are **schematic only**—not
model activations, feature visualizations, probes, or measurements. Phrase them as
possible hierarchies, never as proof that a particular network learned those
features.

### 77–85 min · Connect to deep-learning practice

Trace the image forward pass in order: flatten, one pattern test, 128 hidden
features, class scores, softmax, label-selected loss. Then stack examples into a
minibatch, count both affine layers' parameters, and complete the single numerical
update. Close on the deck's final sentence: training changes the parameters, not
the architecture.

If limited to 70 minutes, shorten the multiclass boundary example and treat the
three-panel interpolant comparison as a quick visual read. Do not cut the
all-four-row XOR verification, the UAT existence caveat, or the
minibatch-to-update ending.

## Board checkpoints

1. **Shapes:** $(m\times d)(d\times1)\to(m\times1)$, then
   $(K\times m)(m\times1)\to(K\times1)$.
2. **Why linear depth collapses:**
   $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$.
3. **Exact XOR table:** calculate $h_1,h_2,z,\hat y$ for all four rows and use
   $\hat y=1$ when $z\ge0$.
4. **One hinge/tent interval:** make the piecewise behavior visible before using
   the phrase “universal approximation.”
5. **One parameter update:** identify the old value, gradient, learning rate, and
   new value; do not let the arithmetic obscure which parameter changed.

## Notebook choreography

### `notebooks/L02/01_linear_regression_gd.ipynb`

- Use after the linear-regression slides or as a pre-class companion.
- The points are seeded **synthetic** observations from $y=2x+1+\epsilon$.
- Run gradient descent, then reveal the closed-form solution and the numerical
  agreement. This is an algorithm check, not real-world validation.
- The loss curve supports the claim that this run converged; it does not establish
  that gradient descent always converges for every objective.

### `notebooks/L02/02_mlp_xor.ipynb`

- Use immediately after the deck's all-four-input XOR table.
- First fit the single linear classifier: its 0.50 accuracy is a seeded optimizer
  result on the four constructed XOR points.
- Then instantiate the **same exact two-hidden-unit weights as the deck** and
  print $x_1,x_2,h_1,h_2,z,\hat y,y$ for all four rows. Accuracy must be 1.00.
- The exact MLP is labelled **constructed + computed**. There is deliberately no
  claim that training learned those MLP weights.
- Finish with the side-by-side probability surfaces: a fitted linear model versus
  the constructed exact MLP.

For either notebook, restart the kernel and run all cells sequentially before
class. A stale output is especially damaging here because the exact network is
also verified numerically in the slides.

## Common wrong turns

- **“A neuron is nonlinear because it has a bias.”** No: affine plus affine still
  collapses. The activation creates the nonlinearity.
- **“Softmax bends the decision boundary.”** No: softmax preserves the winning
  score; ties between linear scores remain flat sets.
- **“The XOR MLP learned those weights.”** Not in this lecture. The construction
  demonstrates representational capacity. Learning comes from an optimizer and
  data, which is a different claim.
- **“Universal approximation means one hidden layer is always enough in
  practice.”** It is an existence statement with no guarantee of compact width,
  easy optimization, finite-sample generalization, or useful inductive bias.
- **“The hierarchy pictures are activation maps.”** They are conceptual
  schematics. Real activation evidence would require a trained model, specified
  input, captured tensors, and a stated visualization method.
- **“A minibatch is a different network.”** It is the same parameters applied to
  multiple rows; only the leading data dimension changes.

## Exit ticket

Give students $(x_1,x_2)=(1,1)$ and ask them to compute both hidden activations,
the logit, and the class for the exact XOR MLP. Then ask one sentence: “What does
universal approximation promise, and what does it not promise?”
