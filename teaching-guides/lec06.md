# Lecture 6 · Making Deep Networks Trainable · Teaching Guide

Audience: undergraduates who have completed Lectures 1–5. Pace from the
104-page handout (one title, seven section canvases, 90 teaching canvases, one
sources page, and five optional canvases) rather than the 209 progressive-
reveal frames in the presentation build.

## The spine

Open by retrieving the central fact from backpropagation: depth multiplies
local derivatives. Strip the network down to a repeated scalar multiplication,
work the forward and backward products, and show how $0.9^{30}$ and
$1.1^{30}$ turn a small per-layer mismatch into vanishing or exploding scale.
Only then ask how the initial weight distribution can make the typical
per-layer gain close to one.

Derive the answer from one neuron. Random initialization has two jobs: break
symmetry and choose a scale. After the symmetry and all-zero examples, stop at
the explicit remaining question: how large should the random weights be? For
the scale calculation, predict the RMS of a 100-term sum, show why unit-RMS
terms produce RMS 10, repair the linear sum with weight standard deviation
0.1, and only then generalize to fan-in. Follow with the backward fan-out sum,
Xavier, the ReLU half-second-moment example, and finally He/Kaiming. The
30-layer spiral is the empirical test of that calculation: data, architecture,
loss, Adam learning rate, seed, and weight directions are fixed while only
$W_\ell\leftarrow\alpha W_\ell$ changes for
$\alpha\in\{0.5,1,1.5\}$.

The lecture is one cumulative argument, not four mini-lectures. Repeated
multiplication creates the problem; initialization controls update-zero scale;
the $\alpha$ experiment checks the prediction; normalization controls scale
using current activations; residual connections add a direct gradient path.
Every mechanism returns to the same fixed spiral experiment.

## Where it sits

- Lecture 2 established affine maps, activations, MLP composition, and the gap
  between expressivity and learnability.
- Lectures 3–4 established Jacobians, local derivative gates, and reverse-mode
  backpropagation.
- Lecture 5 established that the gradient and optimizer update are different
  objects and made Adam the current baseline.
- Lecture 6 asks whether the forward representation and backward sensitivity
  remain usable before the optimizer acts.
- Lecture 7 will ask whether a trainable network generalizes. Keep BatchNorm's
  possible regularization effects subordinate to today's trainability contract.

## Learning contract

By the end, students should be able to:

1. explain examplewise ReLU gating, define a neuron that is dead on a training
   set, and show why cloned or all-zero hidden neurons cannot specialize;
2. distinguish symmetry breaking from scale selection;
3. explain vanishing and exploding activations and gradients as repeated local
   gain errors;
4. use RMS rather than a width-dependent raw norm;
5. derive the fan-in rule, Xavier initialization, and He initialization;
6. interpret BatchNorm and LayerNorm by naming the entries that share
   statistics and by stating train/eval behavior;
7. explain how $h_{\ell+1}=h_\ell+F_\ell(h_\ell)$ creates an identity
   contribution in backpropagation;
8. diagnose a failing deep network from layerwise measurements before stacking
   fixes.

## The fixed experiment

Use the exact experiment implemented in
`lecture6/diagrams/trainability_experiment.py` and the executed notebook
`notebooks/L06/deep_network_trainability_autopsy.ipynb`.

```text
dataset        3-class 2-D spiral; 300 points per class; seed 7
architecture   2 → [Linear(128) → ReLU] × 30 → 3
loss           full-batch cross-entropy
optimizer      Adam, learning rate 3e-4
weight seed    11
base weights   hidden layers: He/Kaiming Normal(0, 2/fan_in); zero biases
only change    W_l ← α W_l, α ∈ {0.5, 1.0, 1.5}
```

The constructor resets the same seed before each model. The output layer also
uses `Normal(0, 2/fan_in)` in this controlled experiment, although no ReLU
follows it; this keeps the common scaling relation exact and is not a general
recommendation for output heads. Verify
`W_0.5 == 0.5 * W_1.0` and `W_1.5 == 1.5 * W_1.0` before interpreting the
curves. The output layer is scaled in the same way as the hidden layers.
Every update uses all 900 examples. This matters later: the BatchNorm repair
uses a fixed full batch, so batch composition does not change from one update
to the next.

The fixed-seed evidence at initialization is:

| $\alpha$ | input activation RMS | layer-30 activation RMS | input-gradient RMS | final active fraction |
|---:|---:|---:|---:|---:|
| 0.5 | 0.419 | $1.02\times10^{-9}$ | $4.57\times10^{-13}$ | 0.493 |
| 1.0 | 0.419 | 1.10 | $1.01\times10^{-3}$ | 0.493 |
| 1.5 | 0.419 | $2.10\times10^5$ | $4.15\times10^2$ | 0.493 |

All three runs are finite at initialization. Positive scaling preserves every
preactivation sign, so the active-fraction traces are identical. This is a key
diagnostic lesson: finite and half-active do not imply a usable scale.

Return to the exact positive-homogeneity calculation at each measurement. With
zero biases, positive $\alpha$, and identical weight directions,

\[
\operatorname{ReLU}(\alpha z)=\alpha\operatorname{ReLU}(z)
\quad\Longrightarrow\quad
h_\ell^{(\alpha)}=\alpha^\ell h_\ell^{(1)}.
\]

Do not jump directly from this per-example statement to the reported RMS.
Stack the 900 examples first and then use the homogeneity of RMS:

\[
H_\ell^{(\alpha)}=\alpha^\ell H_\ell^{(1)},
\qquad
\operatorname{RMS}(H_\ell^{(\alpha)})
=\alpha^\ell\operatorname{RMS}(H_\ell^{(1)}).
\]

Before the symbols, show the two relative-scale tracks:

```text
alpha = 0.5:  1 -> 0.5 -> 0.25 -> 0.125 -> ... -> 9.31e-10
alpha = 1.5:  1 -> 1.5 -> 2.25 -> 3.375 -> ... -> 1.92e5
```

The value $\operatorname{RMS}(H_{30}^{(1)})=1.0969$ is a measured reference
from the $\alpha=1$ forward pass, not a theoretical prediction. It calibrates
the absolute scale; $\alpha^{30}$ predicts the other two runs:

\[
1.0969(0.5)^{30}=1.02\times10^{-9},\qquad
1.0969(1.5)^{30}=2.10\times10^5.
\]

Both predictions match the actual layer-30 tensors to displayed precision.
Keep returning to this worked example: predict the tensor first, inspect the
measurement second, and connect the result back to the He rule already derived.

Before moving from activation scale to the dataset-average loss, hold one
example fixed: sample 132 has
$x=(0.428801,-0.410058)$ and true class $y=0$. The 30 hidden weight matrices
and the output matrix are all scaled, so zero biases and ReLU homogeneity give

\[
z^{(\alpha)}=\alpha^{31}z^{(1)}.
\]

Work softmax for the $\alpha=1$ row on the board, including the stable
subtract-the-maximum step, and then compare all three scales:

| $\alpha$ | logits $(z_0,z_1,z_2)$ | probabilities $(p_0,p_1,p_2)$ | predicted class | this point's CE |
|---:|---|---|---:|---:|
| 0.5 | $(-6.1\times10^{-10},-3.2\times10^{-11},-4.9\times10^{-10})$ | $\approx(0.333,0.333,0.333)$ | 1 | 1.099 |
| 1.0 | $(-1.313,-0.070,-1.047)$ | $(0.173,0.601,0.226)$ | 1 | 1.753 |
| 1.5 | $(-377633,-20019,-301145)$ | $\approx(0,1,0)$ | 1 | 357614 |

Positive scaling preserves $z_1>z_2>z_0$, hence the same argmax, but softmax
responds to the gaps. Larger gaps make the same wrong class-1 prediction more
confident, so $-\log p_0$ grows enormously. The displayed $p_0=0$ in the last
row is float32 underflow/rounding; PyTorch's stable log-sum-exp cross-entropy
is finite. Keep saying “this point's loss”: it is not the 900-example mean
reported in the training table below.

After 150 Adam updates:

| $\alpha$ | initial loss | final loss | final training accuracy | first non-finite |
|---:|---:|---:|---:|---:|
| 0.5 | 1.09861 | 1.09861 | 0.333 | none |
| 1.0 | 1.22422 | $8.07\times10^{-5}$ | 1.000 | none |
| 1.5 | 189359 | 0.0819 | 0.994 | none |

Do not call the $\alpha=1.5$ run a NaN run. It is an *exploded-scale* run that
remains finite in float32 and eventually recovers noisily under Adam. This is
more instructive than manufacturing a non-finite result: optimizer adaptation
can survive a terrible scale without making the scale healthy.

## Exact 80-minute route

| Time | Act | Teaching move |
|---:|---|---|
| 0–10 | why gradients vanish or explode | Draw the repeated deep transformation, work the four-layer scalar forward and backward products, calculate RMS and one per-layer RMS gain from numbers, then compute the 30-layer effect. Connect the gradient reaching layer 1 to its parameter update. |
| 10–34 | initialization from one neuron | Separate symmetry breaking from scale selection. Work one clone update, state the lost-capacity consequence, and compare the XOR decision surfaces. Recall the ReLU gate with positive and negative numbers, distinguish an inactive example from a dead neuron, and then calculate both blocked gradient paths at all-zero initialization. Reopen the scale question explicitly. Predict the 100-term RMS, work the 10-to-1 numerical repair, then derive fan-in, fan-out, Xavier, and He. |
| 34–49 | test on the 30-layer spiral | Reveal runs A–C, define the network tensors, apply the $0.125$ He standard deviation, and introduce the controlled $\alpha$ perturbation. Predict $\alpha^\ell$, check layer 30, run the compact notebook, and interpret activation, probability, loss, and gradient consequences. |
| 49–63 | normalization | Work one feature $[1,2,3,4]$ through centering, standardization, and $\gamma,\beta$; define $Z_\ell\in\mathbb R^{B\times D}$; compare BN columns with LN rows; show batch dependence, eval behavior, placement, and the measured repair. |
| 63–73 | residual route | Ask whether a block can copy $h_\ell$, introduce the same-shape addition, state the limitations of the lecture toy, differentiate a scalar block before the vector Jacobian, then compare the three spiral variants. |
| 73–80 | diagnosis | Work the collapsed trace from symptom to controlled test, assign the different exit trace, and return to all three values of $\alpha$. |

The 86 teaching canvases form the core narrative; the sources page and five
explicitly optional canvases are outside the normal route. The optional pages
hold the unequal-input-unit example, sigmoid tangent, PyTorch initialization
line, and two extra diagnostic cases. If discussion runs long, reduce audience
discussion time on the clone calculation and show only the conclusion of the
drift plot. Preserve the layerwise measurements, the He derivation, the complete
four-number normalization calculation, the scalar-to-vector residual
derivative, and the final diagnosis.

## Teach it like this

### Cold open

Begin with the same deep chain students met in backpropagation. Temporarily set
the activation to the identity and the biases to zero so there is only one
mechanism to inspect: repeated multiplication. Work $h_\ell=qh_{\ell-1}$ with
$q=0.5,1,1.5$ for four layers, then repeat the calculation backward with an
arriving output gradient of one. Generalize only after the numbers are visible.

Before saying “RMS gain,” calculate one. For
$g=(2,-2,1,-1)$, the ordinary mean is zero even though the coordinates are not
small, while
$\operatorname{RMS}(g)=\sqrt{(4+4+1+1)/4}=1.58$. Then compare two adjacent
gradient vectors with
$q_\ell=\operatorname{RMS}(g_{\ell-1})/\operatorname{RMS}(g_\ell)$; the slide's
$0.90/1.00=0.90$ means that one layer reduced a typical coordinate by 10%.
Only after this definition should $q^{30}$ appear.

At depth 30, use $q=0.9,1,1.1$ to show that even a 10% local mismatch gives
$0.042,1,17.45$. Name the consequences precisely: an early layer with a tiny
gradient barely changes; an enormous gradient can create unstable or
non-finite updates. This retrieves Lecture 3 and creates the need for the
initialization calculation. The spiral curves come after He/Kaiming has been
derived; they test a prediction rather than introduce unexplained notation.

### One neuron before layer notation

Use generic symbols on the first pass. Draw $h_1,\ldots,h_n$ entering one
neuron and write

\[
u_i=w_i h_i,
\qquad
z=u_1+\cdots+u_n.
\]

Only two definitions are needed here: a weighted contribution is $u_i$, and
the number of incoming values $n$ is the fan-in. Do not introduce
$\ell,i,j,d_\ell$ before the numerical idea is visible. Present the scale
calculation in this order:

1. state the bridge from the preceding slides: independent nonzero weights
   solve symmetry, but their standard deviation still sets the one-layer gain;
2. ask whether 100 independent, zero-mean, unit-RMS contributions produce a
   sum with RMS closer to 1, 10, or 100;
3. reveal the pattern $n=1\mapsto1$, $n=4\mapsto2$,
   $n=100\mapsto10$, and calculate
   $\mathbb E[z^2]=100$, $\operatorname{RMS}(z)=10$;
4. keep fan-in 100 fixed and compare $\operatorname{Std}(w)=1$ with
   $\operatorname{Std}(w)=0.1$: the latter gives
   $\sqrt{100}(0.1)=1$ for the linear sum;
5. generalize to
   $\operatorname{Var}(w)=1/n$ and
   $\operatorname{Std}(w)=1/\sqrt n$;
6. define fan-out on its own branching diagram, derive the backward
   $1/n_{\text{out}}$ preference, and only then introduce Xavier;
7. use the four-number ReLU example to show that ReLU keeps half the second
   moment, so fan-in 100 changes from 0.1 to
   $\sqrt{2/100}\approx0.141$;
8. apply He/Kaiming to width 128, obtaining 0.125, and then define
   $\alpha=0.5,1,1.5$ as controlled perturbations of the same random
   directions.

The 0.1 result is explicitly for the linear sum before ReLU. This avoids an
apparent contradiction when 0.141 appears two slides later. Distinguish the
two scaling statements aloud:

\[
\operatorname{Std}(w)\propto\frac1{\sqrt n},
\qquad
\operatorname{Var}(w)\propto\frac1n.
\]

When the deck returns to the actual 30-layer MLP, draw the node-level network
and then read the convention aloud:

```text
ell = hidden-layer number
i   = coordinate entering that layer
j   = neuron receiving those coordinates
s   = training example, introduced only when the batch is shown
```

At that point map the generic sum to
$z_{\ell,j}=\sum_i W_{\ell,ji}h_{\ell-1,i}+b_{\ell,j}$ and point out that a
row of $W_\ell$ has fan-in $d_{\ell-1}$ while a column participates in
fan-out $d_\ell$.

Do not lead with $W=\alpha\sqrt{2/\text{fan-in}}\,\epsilon$. By the time that
formula appears, every factor should already have a numerical meaning.

### Symmetry before variance

Draw two hidden neurons with identical incoming *and outgoing* parameters.
Students should
say the chain aloud:

```text
same weights → same preactivation → same activation → same gradient → same update
```

Then distinguish two questions:

- randomness gives neurons different directions;
- variance determines whether the repeated computation has a usable scale.

Do not stop at “the gradients are equal.” State the capacity consequence. If
all four hidden units have the same incoming weights, bias, and outgoing
coefficient, then

\[
h_1(x)=\cdots=h_4(x)=h(x),\qquad
s(x)=\sum_{j=1}^4v h_j(x)+c=4v h(x)+c.
\]

The layer contains four parameterized neurons but only one distinct hidden
feature. The output can rescale that feature; it cannot manufacture three new
directions. Use the XOR comparison immediately afterward. The controlled
model is `Linear(2,4) → ReLU → Linear(4,1)` on four Gaussian blobs at
$(\pm1,\pm1)$. With cloned parameters its hidden activation matrix stays rank
one and one blob remains on the wrong side of the boundary; independent
initialization lets the hidden units specialize and separate all four blobs.
Ask students to identify the overlapping hidden hinge lines before revealing
the accuracy.

The browser lab linked from the slide has three reset modes: exactly cloned,
slightly perturbed, and independent. Use one-step mode first. In the cloned
run, point out that the maximum separation between corresponding hidden
parameters remains zero even though they are separate parameters. Then reset
to the perturbed mode and show the units begin to diverge. The lesson is lost
nonlinear capacity, not “the network cannot learn anything”: the cloned model
can still learn one useful linear split and typically reaches 75% on the four
balanced blobs.

Keep the two reproducible implementations distinct when quoting numbers. The
static slide comes from `lecture6/diagrams/symmetry_experiment.py`: 320 Gaussian
points, seed 7, PyTorch Adam at 0.03 for 300 full-batch updates. Its audited
results are cloned 75%, BCE 0.47758, rank 1 versus independent 100%, BCE
0.00091, rank 4; the exact configuration and final tensors are recorded in
`lecture6/evidence/symmetry_experiment.json`. The browser lab is a lighter
teaching model: a deterministic 100-point XOR grid and full-batch gradient
descent at 0.20. Use it to inspect invariance step by step, not as a second
claim about the PyTorch optimizer trace.

Keep the qualification precise: equal incoming rows alone do not guarantee the
symmetry. Hidden biases and outgoing coefficients must also match. Ordinary
full-batch or minibatch gradient descent preserves exact clones; dropout or
independent noise can break them. Do not implement this demonstration with a
single shared parameter, because gradient accumulation and Adam state then
describe weight tying rather than cloned initialization.

### ReLU death before all-zero initialization

Do not jump from the clone example directly to $\operatorname{ReLU}'(0)$.
Retrieve the gate first:

\[
z=w^\top x+b,\qquad h=\operatorname{ReLU}(z),\qquad
\frac{\partial\mathcal L}{\partial z}
=\frac{\partial\mathcal L}{\partial h}\mathbf 1[z>0].
\]

On the first new slide, use the same arriving gradient, $\partial\mathcal
L/\partial h=4$, for both numerical cases. With $x=2,w=-1,b=-1$, the
preactivation is $-3$, so the gate is closed and the gradients for $w$ and $b$
are zero. With $x=2,w=1,b=-1$, the preactivation is $1$, so the gate is open,
$\partial\mathcal L/\partial w=8$, and $\partial\mathcal L/\partial b=4$.
Keep saying *preactivation*: a ReLU activation itself is never negative.

Then define death over data rather than over one example. For
$x\in\{-1,0,1\}$, $w=1$, and $b=-0.2$, two examples are inactive but the
third opens the gate; the neuron can still receive a data gradient. With
$b=-2$, all three preactivations are strictly negative and every example's
incoming-gradient contribution is zero. Call this “dead on this training set”
or “dead for this fixed representation.” An inactive minibatch alone does not
establish dataset-wide death, and a deeper unit can sometimes revive if an
earlier layer changes its inputs.

Only then move to zero initialization. At $z=0$ the ReLU derivative is
mathematically undefined; PyTorch chooses zero. For the complete two-layer
zero initialization, however, do not blame only that convention. The two
blocked paths are

\[
\frac{\partial\mathcal L}{\partial v_i}
=(\hat y-y)h_i=0
\quad\text{because }h_i=0,
\]

\[
\frac{\partial\mathcal L}{\partial w_i}
=(\hat y-y)v_i\operatorname{ReLU}'(0)x=0
\quad\text{because }v_i=0
\text{ and PyTorch closes the gate at zero}.
\]

Finish the update: $w_i^+=v_i^+=0$, so the next step starts from exactly the
same state. Contrast it explicitly with the previous failure: nonzero clones
can move but learn one shared feature; all-zero units do not begin learning a
hidden feature. Zero biases remain perfectly compatible with He/Kaiming when
the weights are independently random. If an output bias exists, it may learn a
constant or class base rate while the hidden features stay frozen.

For caveats in discussion: fresh SGD or Adam state cannot escape an exact zero
data gradient; accumulated momentum, regularization on nonzero parameters,
noise, or reinitialization can alter a unit later. Leaky ReLU passes a
negative-side gradient, but it still does not rescue a completely zero
two-layer network when the outgoing weights and hidden activations are both
zero.

### Depth and gates

Use the actual network depth before introducing matrix language:
$0.9^{30}\approx0.042$ and $1.1^{30}\approx17.45$. The scalar example earns
the qualitative claim; singular values are only an intuitive note.

For an activation, keep both roles visible:

\[
h=\phi(z),\qquad \bar z=\phi'(z)\bar h.
\]

The sigmoid calculation is now a backup slide, not part of the 80-minute core.
If students need the contrast, at $z=5$, $\sigma'(5)\approx0.00665$, so ten
such gates contribute about $1.7\times10^{-22}$. State the qualification:
sigmoid is appropriate for a binary output probability; the failure is
repeating saturated gates inside a deep hidden stack. The core route stays
with the actual ReLU model: one side passes and one side blocks.

The active-fraction reveal should surprise students: all three $\alpha$ traces
are exactly the same because positive scaling does not change signs. Ask what
additional measurement is needed; the answer is activation and gradient RMS.

## Reproducible derivation: Xavier and He

Track raw second moments because ReLU outputs are not zero-mean.

For one preactivation,

\[
z=\sum_{i=1}^{n_{\text{in}}}w_i h_i.
\]

First fix 100 incoming coordinates with RMS one, so
$\sum_i h_i^2=100$. Across independent zero-mean initializations with
$\operatorname{Std}(w)=1$,

\[
\mathbb E_W[z\mid h]=0,
\qquad
\operatorname{Var}_W(z\mid h)
=\sum_i h_i^2\operatorname{Var}(w_i)=100,
\]

and therefore $\operatorname{Std}_W(z\mid h)=10$. Repeating the same
calculation with $\operatorname{Std}(w)=0.1$ gives variance one and standard
deviation one. The statement concerns a distribution over random
initializations; one realized preactivation need not equal 10.

Under independent zero-mean weights, weights independent of activations, and
comparable coordinate second moments,

\[
\mathbb E[z^2]
=n_{\text{in}}\operatorname{Var}(w)\mathbb E[h^2].
\]

The cross terms vanish because $\mathbb E[w_iw_j]=0$ for $i\ne j$. Forward
preservation therefore prefers
$\operatorname{Var}(w)\approx1/n_{\text{in}}$, equivalently
$\operatorname{Std}(w)\approx1/\sqrt{n_{\text{in}}}$. Use
$\mathbb E[h^2]$, not $\operatorname{Var}(h)$: after ReLU the activation has a
positive mean, so those quantities differ.

Backward propagation gives

\[
\bar h_{\ell-1}=W_\ell^\top\bar z_\ell,
\]

so the matching approximation prefers $1/n_{\text{out}}$. Xavier balances the
two:

\[
\operatorname{Var}(W)=\frac{2}{n_{\text{in}}+n_{\text{out}}}.
\]

For a symmetric preactivation,

\[
\mathbb E[\operatorname{ReLU}(z)^2]=\frac12\mathbb E[z^2].
\]

Matching this gate requires

\[
\frac12 n_{\text{in}}\operatorname{Var}(w)\approx1
\quad\Longrightarrow\quad
\operatorname{Var}(W)=\frac{2}{n_{\text{in}}}.
\]

For 100 ReLU inputs, $\operatorname{Std}(W)=\sqrt{2/100}\approx0.141$;
equal-width Xavier gives 0.1. State the limitation precisely: these rules aim
for a sensible second-moment scale at initialization under simplifying
assumptions; they do not keep the variance exactly one throughout training.

Immediately put the formula back into the fixed model. The $2\to128$ stem has He
standard deviation $\sqrt{2/2}=1$; every $128\to128$ hidden map has
$\sqrt{2/128}=0.125$. Multiplying the shared samples by
$\alpha\in\{0.5,1,1.5\}$ produces $0.0625,0.125,0.1875$, hence expected
per-layer RMS multipliers of roughly $0.5,1,1.5$. These are the multipliers
seen in the measured $\alpha^{30}$ traces.

## Live normalization calculation

Start from the drift measurement: He initialization chooses a scale at update
0, but after 20 updates the weights and preactivation distribution have moved.
Normalization recomputes a reference from the current preactivations.

Use $Z_\ell\in\mathbb R^{B\times D}$ for the preactivation matrix before
ReLU. Rows are examples and columns are hidden features. Treat
$z_{\cdot d}=[1,2,3,4]$ as one selected feature across four examples. Work all
four steps rather than jumping to the final formula.

1. Identify the set: four values of feature $d$, one from each example.
2. Subtract their mean:

   \[
   \mu_d=2.5,
   \qquad z_{\cdot d}-\mu_d=[-1.5,-0.5,0.5,1.5].
   \]

3. Compute the divisor-$B$ variance and standardize:

   \[
   \sigma_d^2
   =\frac{(-1.5)^2+(-0.5)^2+0.5^2+1.5^2}{4}=1.25,
   \]

   \[
   \hat z_{\cdot d}
   =\frac{z_{\cdot d}-\mu_d}{\sqrt{\sigma_d^2}}
   \approx[-1.342,-0.447,0.447,1.342].
   \]

   Check that the standardized values have mean zero and RMS one. BatchNorm
   computes a variance; RMS is only the lecture's reporting metric. They agree
   here after centering. In code the denominator is
   $\sqrt{\sigma_d^2+\epsilon}$.

4. Apply the learned affine transformation. Starting values
   $\gamma_d=1,\beta_d=0$ leave the standardized feature unchanged. For a
   numerical example, use $\gamma_d=2,\beta_d=1$:

   \[
   y_{\cdot d}=2\hat z_{\cdot d}+1
   \approx[-1.683,0.106,1.894,3.683].
   \]

Only after this calculation introduce a general selected set $S$:

\[
\mu_S=\frac{1}{|S|}\sum_{i\in S}z_i,\qquad
\sigma_S^2=\frac{1}{|S|}\sum_{i\in S}(z_i-\mu_S)^2,
\]

\[
\hat z_i=\frac{z_i-\mu_S}{\sqrt{\sigma_S^2+\epsilon}},\qquad
y_i=\gamma_i\hat z_i+\beta_i.
\]

Then ask which entries belong to $S$. BatchNorm selects one feature column of
$Z_\ell$ across examples. LayerNorm selects one example row across features.
For the matrix shown in the deck, BatchNorm repeats the complete
$[1,2,3,4]$ calculation for the first column; LayerNorm instead standardizes
the first row $[1,2,6]$ to approximately
$[-0.926,-0.463,1.389]$.

Make BatchNorm's training-time batch dependence numerical. Hold the raw value
$z=1$ fixed:

\[
1\text{ in }[1,2,3,4]\longmapsto-1.342,
\qquad
1\text{ in }[-2,-1,0,1]\longmapsto+1.342.
\]

The raw value did not change; only the other examples in its batch did. During evaluation,
BatchNorm uses running mean and variance accumulated during training. LayerNorm
uses the same per-example operation in training and evaluation. Do not imply
that BatchNorm forbids batched inference.

Place the operation explicitly in the model:

```text
H_(l-1) → Linear → Z_l → BatchNorm → ReLU → H_l
```

In code this is `h = relu(batch_norm(linear(h)))`, so statistics are computed
from preactivations, not post-ReLU activations.

Before revealing the BatchNorm repair, predict its ReLU output RMS:

\[
\mathbb E[\hat z^2]\approx1
\quad\Longrightarrow\quad
\operatorname{RMS}(\operatorname{ReLU}(\hat z))\approx\sqrt{1/2}=0.707.
\]

The measured layer-30 value is 0.713. Connect this prediction to the earlier He
calculation so normalization continues the same scale analysis.

For $Z_\ell\in\mathbb R^{B\times D}$:

| Question | BatchNorm | LayerNorm |
|---|---|---|
| shared statistics | one feature column across examples | one example row across features |
| another batch member matters in training | yes | no |
| evaluation | stored running statistics | same per-example operation |
| CNN interpretation | one channel over $B,H,W$ | not the usual CNN default |

In this repair experiment, $B=900$: every training update contains the entire
spiral dataset. The composition of the batch is therefore identical across
updates. This is a deliberately controlled demonstration of normalization's
scale effect, not an experiment on mini-batch noise. Do not teach “internal
covariate shift” as a complete settled explanation; keep the claim tied to the
measured activation and gradient scales.

## Residual numeric

Start from

\[
h_{\ell+1}=h_\ell+F_\ell(h_\ell).
\]

First establish the representation question with the scalar identity task:
$h_\ell=3$ and target $y=3$. A plain scalar layer $\hat y=wh_\ell$ must learn
$w=1$; the residual parameterization $\hat y=h_\ell+wh_\ell$ already copies
the input when $w=0$. Then show the vector addition
$[2,-1]^\top+[0.3,0.2]^\top=[2.3,-0.8]^\top$ and state that the two terms must
have the same shape.

Differentiate one scalar block before showing a Jacobian. Let the arriving
gradient be $g_{\ell+1}=\partial\mathcal L/\partial h_{\ell+1}=4$. For an
abstract local derivative $F'_\ell(h_\ell)=-0.01$,

\[
g_\ell=g_{\ell+1}\bigl(1+F'_\ell(h_\ell)\bigr)
=4(1-0.01)=3.96.
\]

Label $-0.01$ as an illustrative derivative, not a measurement from the
spiral model. Once the scalar rule is clear, define
$J_{F_\ell}=\partial F_\ell(h_\ell)/\partial h_\ell\in\mathbb R^{D\times D}$
and column gradients $g_\ell=\partial\mathcal L/\partial h_\ell$:

\[
g_\ell=(I+J_{F_\ell})^\top g_{\ell+1}
=g_{\ell+1}+J_{F_\ell}^\top g_{\ell+1}.
\]

Continue the same abstract derivative through 20 blocks:

\[
(-0.01)^{20}=10^{-40}\quad\text{for a branch-only stack},
\qquad
|1+F'|^{20}=0.99^{20}\approx0.818.
\]

These are different parameterizations and therefore different initial maps;
the comparison isolates the algebraic effect of the identity term. The lecture
experiment first maps $\mathbb R^2\to\mathbb R^{128}$ with a stem, then uses 29
matching-width blocks

\[
h_{\ell+1}=h_\ell+\operatorname{ReLU}(W_\ell h_\ell).
\]

This branch is deliberately minimal. It is not a standard modern ResNet block,
and the experiment does not initialize the whole model to the exact identity.
Because the branch output is a ReLU, it adds only nonnegative coordinate
corrections. Its purpose is to isolate the direct $h_\ell$ route. State the
remaining caveats: the branch can still explode; $I$ and $J_F$ can cancel along
a direction; and mismatched shapes require a projection.

## Repair experiment

Return to the collapsed $\alpha=0.5$ case. Keep data, depth, width, loss, Adam,
learning rate, and seed fixed. Compare:

1. plain `Linear → ReLU`;
2. `Linear → BatchNorm → ReLU`;
3. one input stem followed by the lecture toy
   `h + ReLU(Linear(h))` blocks.

Before training, show the measurements that distinguish the mechanisms:

| route | layer-30 activation RMS | input-gradient RMS | initial loss |
|---|---:|---:|---:|
| plain | $1.02\times10^{-9}$ | $4.57\times10^{-13}$ | 1.09861 |
| BatchNorm | 0.713 | 0.225 | 1.11666 |
| residual toy | 698.57 | 0.329 | 177.84 |

After 150 updates:

| route | initial loss | final loss | final accuracy |
|---|---:|---:|---:|
| plain | 1.09861 | 1.09861 | 0.333 |
| BatchNorm | 1.11666 | 0.00550 | 1.000 |
| residual toy | 177.84 | 0.2434 | 0.986 |

The residual toy begins with very large activations and loss because each
positive branch output is added to the state. Its usable input-gradient RMS
shows what the direct term changes, while the large activation RMS shows what
it does not fix. This comparison demonstrates mechanisms on one controlled
case; it is not a universal ranking of architectures.

## Notebook choreography

### Minimal scale-through-depth notebook

Use `notebooks/L06/01_activation_scale_through_depth.ipynb` immediately after
the $\alpha^\ell$ prediction. This is the small executable version shown in
the deck. Its `forward()` method explicitly returns the 30 post-ReLU tensors,
so students see exactly where each RMS value comes from without first learning
hooks.

The notebook keeps the spiral, 30-by-128 ReLU architecture, seed 11, zero
biases, and common weight directions fixed. It then:

1. verifies all 31 weight matrices are scaled copies of the $\alpha=1$ run;
2. checks the complete tensor identity
   $H_\ell^{(\alpha)}=\alpha^\ell H_\ell^{(1)}$ at every hidden layer;
3. plots activation RMS through depth;
4. measures logit RMS, softmax confidence, entropy, cross-entropy, and accuracy
   before any optimizer step;
5. runs one backward pass and plots gradient RMS from $H_{30}$ back to $X$.

Pause before each output cell and ask for a prediction. Emphasize that the
input-gradient RMS is a propagation diagnostic, not a parameter update, and
that the $\alpha=1.5$ run is badly scaled but finite. The full autopsy notebook
adds training, normalization, and residual comparisons later.

### PyTorch hooks tutorial

Use `notebooks/L06/00_pytorch_hooks_for_layerwise_diagnostics.ipynb` as a
short prerequisite or follow-up tutorial. The main deck should teach only the
conceptual boundary:

```text
h_(l-1) → Linear → z_l → ReLU → h_l
                   ↑              ↑
             Linear hook     ReLU hook
```

A hook on `Linear` records the preactivation $z_\ell$; a hook on `ReLU`
records the post-ReLU activation $h_\ell$ used by the layerwise RMS trace. The
notebook then works from one deterministic layer to a complete 30-layer probe:

1. compute $z_1$ and $h_1$ manually;
2. register and remove forward hooks;
3. verify the Linear-versus-ReLU distinction numerically;
4. explain non-leaf gradients and `retain_grad()`;
5. record $\partial\mathcal L/\partial h_\ell$ with a tensor gradient hook;
6. contrast tensor hooks with `register_full_backward_hook()`;
7. store detached scalars, clear records, and remove handles;
8. collect activation RMS, active fraction, finite status, and gradient RMS
   across a 30-layer MLP.

Keep hooks out of the initialization derivation. They are an implementation
tool for observing the tensors already defined by the mathematics. If a model
is under your control and only a few intermediates are needed, returning them
explicitly from `forward()` is often clearer.

### Trainability experiment

The executed `deep_network_trainability_autopsy.ipynb` is one follow-along
sequence:

1. generate and inspect the spiral;
2. define the 30-layer model and the RMS helper;
3. verify that the three models share weight directions;
4. predict $h_{30}^{(\alpha)}=\alpha^{30}h_{30}^{(1)}$ and verify it numerically;
5. inspect activation histograms at layers 1, 10, 20, and 30;
6. reveal activation, gradient, active-fraction, and finite traces;
7. train all three $\alpha$ values with the same Adam configuration;
8. inspect decision regions after 50 updates;
9. show that the healthy initialization drifts after 20 updates;
10. compare the collapsed $\alpha=0.5$ model with BatchNorm and the deliberately
    minimal residual toy;
11. write symptom → measurement → smallest controlled test.

The notebook retains outputs and should run in under one minute on an ordinary
CPU. The deck figures are regenerated by the separate evidence script so slide
builds do not depend on notebook state.

## Misconceptions to intercept

- **“Adam fixes vanishing gradients.”** Adam adapts the update from the gradient
  that arrives. In this run the collapsed gradients are far below Adam's
  default $\epsilon$, so its actual early-layer updates are operationally
  negligible. State the measured mechanism rather than claiming that no
  optimizer could ever revive the model.
- **“Zero initialization only makes learning slow.”** Identical hidden units
  remain identical. When hidden and outgoing weights are all zero, hidden
  activations and both sets of weight gradients are zero, so the same state
  repeats.
- **“A negative preactivation means the ReLU is dead.”** A closed gate for one
  example is ordinary ReLU behavior. Call the neuron dead on the training set
  only when no training example opens it.
- **“One hundred unit-RMS terms produce an RMS of 100.”** For independent,
  zero-mean terms, second moments add: $\mathbb E[z^2]=100$, so the RMS is
  $\sqrt{100}=10$.
- **“Variance scales as $1/\sqrt n$.”** Standard deviation scales as
  $1/\sqrt n$; variance is its square and scales as $1/n$.
- **“The 0.1 linear rule contradicts the 0.141 He rule.”** With fan-in 100,
  0.1 preserves the linear sum. A following ReLU removes half the second
  moment, so He uses $\sqrt{2/100}\approx0.141$.
- **“He keeps variance exactly one.”** It matches an approximate second-moment
  gain at initialization under simplifying assumptions.
- **“Half the ReLUs are active, so the network is healthy.”** The three
  $\alpha$ runs have identical active fractions and radically different RMS.
- **“BatchNorm forbids batched inference.”** Evaluation uses stored statistics;
  the batch may still contain many examples.
- **“BatchNorm's complete explanation is internal covariate shift.”** Prefer
  operational claims about scale robustness, conditioning, parameterization,
  and training-time batch noise.
- **“Residuals guarantee non-vanishing gradients.”** The identity contribution
  helps, but the branch may explode or cancel it along a direction.
- **“A non-finite check passed, so the run is healthy.”** The $\alpha=1.5$ model
  is finite while already $2\times10^5$ in layer-30 activation RMS.

## If short on time

Never cut:

1. the scalar forward/backward product and its 30-layer consequence;
2. the negative-versus-positive ReLU example, the definition of dataset-wide
   death, and the two blocked all-zero gradient paths;
3. the 100-term RMS prediction, the 0.1 linear repair, the variance-versus-
   standard-deviation distinction, and the Xavier/He application to the
   128-wide model;
4. the controlled $\alpha$ prediction, the compact notebook, and the measured
   activation/gradient consequences;
5. all four steps of the $[1,2,3,4]$ normalization example, the BN/LN axis
   distinction, `Linear → BatchNorm → ReLU`, and train/eval distinction;
6. the scalar residual derivative before the vector Jacobian and the caveat
   that the lecture branch is only an isolation toy;
7. the final spiral comparison and symptom → suspect → test exit ticket.

Cut in this order:

1. every slide after “Primary sources and reproducible evidence” — they are backups;
2. detailed discussion of the initialization histograms after students have
   seen the RMS traces;
3. the full symmetry derivation after showing one complete clone update;
4. the full repair-curve discussion—show the initialization table and final
   decision regions instead.

For a 45-minute route: repeated-product motivation 6 min; symmetry, the compact
ReLU/death sequence, and the numerical fan-in/Xavier/He derivation 15 min;
controlled $\alpha$ prediction and measured traces 6 min; the complete normalization
example, axes, placement, and evaluation behavior 7 min; scalar-to-vector
residual derivation and the three-way initialization table 7 min; diagnosis
and return to the spiral 4 min. State the drift observation verbally and leave
all optional pages out.
