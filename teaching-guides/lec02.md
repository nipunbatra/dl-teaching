# Lecture 2 · From Linear Models to Neural Networks · Teaching Guide

*Instructor companion for the current public Lecture 2 deck.*

## The spine

Lecture 1 chose losses from probability; Lecture 2 expands the prediction rule
$f_\theta(x)$. Begin with one affine score, show exactly why affine layers alone
cannot represent XOR, and then build and verify one two-unit ReLU MLP. Change the
dataset from four Boolean corners to filled XOR regions so that students learn to
separate a model construction from a broader task claim. ReLU hinges then lead to
universal approximation, followed by one concise account of depth as composition.
End with the handoff

$$
\text{representation}\rightarrow\text{loss}\rightarrow
\text{backpropagation}\rightarrow\text{optimizer}.
$$

## What students should leave able to do

- Track the dimensions of a weighted sum, a data-matrix multiplication, and a
  two-layer MLP.
- Separate the affine pre-activation $u$ from the activation $h=\phi(u)$.
- Explain why stacking affine layers still gives an affine map and why XOR
  defeats a single straight decision boundary.
- Evaluate the deck's exact two-ReLU XOR network on all four Boolean inputs.
- Explain why that same rule becomes a diagonal band over a filled square, and
  distinguish its 75% area agreement from its 4/4 corner result.
- Read the four-ReLU regional XOR rule as a second, task-specific construction.
- State the universal-approximation claim without turning existence into a claim
  about learnability or generalization.
- Explain width as parallel feature variety and depth as sequential composition.
- Count the trainable scalars in a two-layer MLP using
  $m(d+1)+K(m+1)$.
- Connect a richer representation to the familiar loss, then identify
  backpropagation and optimization as the next two steps.

## Evidence language for this lecture

Use the deck tags as route markers: `V` is a visual explanation, `D` is a worked
derivation, `Q/A` is a prediction-and-reveal pair, `I` is an interactive, and
`OPT` is optional. They do **not** by themselves imply empirical evidence.

| Material | Honest evidence label | How to say it |
|---|---|---|
| Weighted-sum, matrix, neuron, XOR, loss, and parameter-count arithmetic | **Computed from displayed values** | “Let us verify the numbers.” |
| Four Boolean XOR inputs and the exact two-unit weights | **Constructed teaching example** | “These weights are chosen to fit the four truth-table points; an optimizer did not discover them here.” |
| Filled-square 75% check and exact four-ReLU regional rule | **Constructed + computed** | “We changed the dataset from four corners to four filled regions, then checked both fixed rules on a dense grid.” |
| Linear-regression notebook data | **Synthetic data; computed fit** | “This checks gradient descent against the closed-form solution on seeded synthetic data.” |
| Universal approximation | **Mathematical representation result** | “Approximating weights exist; the theorem does not promise that training finds them.” |
| Decision-region, node, and vision-hierarchy graphics | **Schematic** | “This diagram explains structure; it is not a measured model trace.” |

Do not call either XOR construction a learned result, silently transfer the
two-ReLU 4/4 claim to the filled-square dataset, call the synthetic regression a
real-data experiment, or treat the vision hierarchy as evidence of what a
particular trained model represents.

## 80-minute route through the current deck

### 0–5 min · Retrieve Lecture 1 and change only the prediction rule

Use “Lecture 1 chose the loss; Lecture 2 expands the prediction rule” as the
handoff. Keep $\hat y=f_\theta(x)$ visible. The loss can remain unchanged while
the function class becomes richer.

### 5–15 min · Linear starting point

Work from one weighted-sum node to the labelled data matrix, one batched matrix
multiplication, and the bias-as-ones column. Make students say the shapes aloud.
Close with the regression geometry: one affine rule can tilt and shift a line or
plane, but it cannot bend. The deck links the
[Linear GD Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L02/01_linear_regression_gd.ipynb)
on this slide; use it as a pre-class check or a short live follow-up rather than
interrupting the conceptual route.

### 15–25 min · From a score to a neuron

Separate the two stages: $u=w^Tx+b$, then $h=\phi(u)$. Compare bounded switches
with ReLU/GELU ramps, and use the saturation calculation to make “locally flat”
numerical. Ask the linear-layers question before revealing that two affine maps
collapse into one. Open the activation explorer only after the algebraic point
is secure.

### 25–35 min · Where one weighted sum fails

Draw the binary straight boundary and let students try to separate XOR. Then
connect the same geometry to multiclass scores: score ties lie on flat sets, and
only the top-scoring portion of a tie is a visible decision boundary. The five
explicit inputs are a calculation, not a benchmark. If time is tight, shorten
the numerical multiclass example; preserve the straight-boundary idea.

### 35–52 min · Build exact four-point XOR, then change the task

Keep the same network throughout:

$$
h_1=\operatorname{ReLU}(x_1+x_2),\qquad
h_2=\operatorname{ReLU}(x_1+x_2-1),\qquad
z=2h_1-4h_2-1.
$$

Verify all four rows. Derive the two hidden functions before substituting one
input, then compute the output probability and loss. The hidden-space coordinate
slide is the payoff: the nonlinear map creates a representation that one output
node can separate. At the architecture slide, pause on
$m(d+1)+K(m+1)$: each neuron contributes its incoming weights plus one bias.

Next, make the scope change explicit: fill the square with quadrant-XOR points.
The same logit depends only on $x_1+x_2$, so it paints the diagonal band
$0.5\le x_1+x_2\le1.5$ and agrees on 75% of the square even though it retains
4/4 on the four corners. The deck links the
[XOR Colab](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L02/02_mlp_xor.ipynb)
at this comparison.

Open the
[ReLU Classification Lab](https://nipunbatra.github.io/interactive-articles/mlp-decision-boundary/).
Compare four XOR corners with filled XOR while holding the rule fixed. Then
compare the two-ReLU corner rule with the four-ReLU regional construction. Use
the logit map, signed hidden-unit contributions, and rotatable 3-D surface to ask
what function the network represents. Reset before training so “constructed
rule” and “optimizer run” remain separate claims.

### 52–67 min · Universal approximation via hinges

Frame one precise question, build shifted ReLU hinges, and assemble the tent
function interval by interval. The 2-D fold and piecewise-linear classification
slides extend the same idea beyond a scalar input without starting a separate
technical detour.

For the approximation claim, fix the target and compact region first, define one
pointwise vertical gap, and only then explain `sup` as the worst gap over the
whole region. The three-panel interpolant comparison is **constructed, not
trained**. Read the theorem in quantifier order and dwell on the warning:
existence is not learnability, data efficiency, compactness, or generalization.

Use the
[ReLU Approximation Lab](https://nipunbatra.github.io/interactive-articles/relu-function-approximation/)
to keep construction and training distinct. In **Construction**, move knots or
change width while watching the function, residual, maximum grid gap, and signed
ReLU terms. In **Training**, hold the target and width fixed so students can see
what one initialization and optimizer run actually finds.

### 67–76 min · Width versus depth

Use the node view first: width adds parallel features; depth passes one learned
representation into the next. The two composition slides make that statement
concrete:

$$
h_1(x)=\operatorname{ReLU}(2x+1),\qquad
h_2(x)=\operatorname{ReLU}(2-h_1(x)).
$$

Read the two axes carefully—layer 2 receives $h_1$, not raw $x$—then combine the
maps into the three input regions. The vision slide is the only hierarchy
example retained: strokes may combine into loops, which may support digit
evidence. Call it a conceptual schematic, not an activation visualization.

### 76–80 min · From representation to learning

Use the single closing diagram. Hidden layers produce features and class scores;
Lecture 1 supplies the loss. Lecture 3 will compute how each parameter affected
that loss, and the optimizer will use those gradients to change the parameters.
Keep this to one slide: its purpose is a clean transition to backpropagation.

If limited to 70 minutes, shorten the multiclass numerical example and the
constructed-interpolant comparison. Do not cut the all-four-row XOR check, the
four-points-versus-filled-regions distinction, or the UAT existence caveat.

## Board checkpoints

1. **Shapes:** $(m\times d)(d\times1)\to(m\times1)$, then
   $(K\times m)(m\times1)\to(K\times1)$.
2. **Why affine depth collapses:**
   $W_2(W_1x+b_1)+b_2=(W_2W_1)x+(W_2b_1+b_2)$.
3. **Exact XOR table:** calculate $h_1,h_2,z,\hat y$ for all four rows and use
   $\hat y=1$ when $z\ge0$.
4. **XOR scope check:** draw the two-ReLU diagonal band over the filled square,
   then contrast it with the four-ReLU quadrant rule.
5. **One hinge/tent interval:** make the slope change visible before using the
   phrase “universal approximation.”
6. **Parameter count:** identify $md+m$ parameters in the hidden layer and
   $Km+K$ in the output layer.
7. **Learning handoff:** representation $\to$ loss $\to$ backpropagation
   $\to$ optimizer.

## Notebook and interactive choreography

### `notebooks/L02/01_linear_regression_gd.ipynb`

- Use after the linear-regression slides or as a pre-class companion.
- The points are seeded **synthetic** observations from $y=2x+1+\epsilon$.
- Run gradient descent, then reveal the closed-form solution and their numerical
  agreement. This is an algorithm check, not real-world validation.
- The deck's “flat trend” slide contains the direct Colab link.

### `notebooks/L02/02_mlp_xor.ipynb`

- Use immediately after the four-points-versus-filled-regions comparison.
- Instantiate the **same exact two-hidden-unit weights as the deck** and print
  $x_1,x_2,h_1,h_2,z,\hat y,y$ for all four rows. Agreement must be 1.00.
- Extend that fixed rule to a dense filled-square XOR grid and verify 0.750
  agreement.
- Construct the four-ReLU regional rule and verify 1.000 agreement away from the
  boundary axes.
- Finish with the three-panel comparison: four corners, the two-ReLU diagonal
  band, and the four-ReLU filled-region rule.
- These fixed networks are labelled **constructed + computed**. They are not
  optimizer results. The deck's scope-comparison slide contains the Colab link.

### Interactive links in the deck

- [Activation explorer](https://nipunbatra.github.io/dl-teaching/interactives/activation-explorer.html): compare shape and saturation.
- [ReLU Classification Lab](https://nipunbatra.github.io/interactive-articles/mlp-decision-boundary/): compare datasets, rules, learned boundaries, hidden contributions, and the 3-D logit surface.
- [ReLU Approximation Lab](https://nipunbatra.github.io/interactive-articles/relu-function-approximation/): compare explicit construction with optimization.

Restart each notebook and run all cells sequentially before class. A stale output
is especially damaging here because the fixed networks are also verified in the
slides.

## Common wrong turns

- **“A neuron is nonlinear because it has a bias.”** No: affine after affine
  still collapses. The activation creates the nonlinearity.
- **“Softmax bends the decision boundary.”** No: softmax preserves the winning
  score; ties between linear scores remain flat sets.
- **“The XOR MLP learned those weights.”** Not in this lecture. The construction
  demonstrates representational capacity; an optimizer run is a different claim.
- **“Two ReLUs solve every dataset called XOR.”** They solve the four Boolean
  corners used in the deck. On filled quadrant XOR, the same rule paints a
  diagonal band and reaches 75% area agreement.
- **“Universal approximation means one hidden layer is always enough in
  practice.”** It is an existence statement with no guarantee of compact width,
  easy optimization, finite-sample generalization, or useful inductive bias.
- **“The vision hierarchy is a measured activation trace.”** It is a conceptual
  schematic. Real activation evidence requires a trained model, a specified
  input, captured tensors, and a stated visualization method.
- **“Depth just means more parameters.”** Depth means sequential composition;
  width adds more features at the same stage.

## Exit ticket

Give students $(x_1,x_2)=(1,1)$ and ask them to compute both hidden activations,
the logit, and the class for the exact XOR MLP. Then ask: “Why does 4/4 on the
truth table not imply perfect classification of filled quadrant XOR?” Close with
one sentence on what universal approximation promises—and what it does not.
