# Lecture 6 · Making Deep Networks Trainable · Teaching Guide
*First-time-instructor companion. Audience: undergraduates who have completed the public Lectures 1–5.*

## The spine (say this in two sentences)

A deep network repeats the same local operation, so a modest scale error or an unhelpful derivative is multiplied many times: signals can disappear or explode on the way forward, and gradients can do the same on the way back. Keep two layerwise gauges visible while adding four complementary controls—activation, initialization, normalization, and a residual path—and make students predict every intervention on one width-4, five-layer ReLU network before showing a trace.

## Where it sits

**Prerequisites in the public sequence:**

- **Lecture 2 · From Linear Models to Neural Networks:** an MLP composes affine maps and nonlinearities; depth means repeating that composition.
- **Lecture 3 · Calculus Toolkit:** gradients, Jacobians, the chain rule, and transposes for column-gradient notation.
- **Lecture 4 · Computation Graphs & Backpropagation:** backprop supplies the gradients by multiplying local Jacobians and accumulating contributions along multiple paths.
- **Lecture 5 · Optimization for Deep Learning:** the optimizer turns those measured gradients into parameter updates, using direction memory, scale memory, and a learning-rate clock.

This lecture asks an earlier architectural question: **do useful forward signals and backward gradients survive the trip through depth in the first place?** Lecture 4 supplies gradients; Lecture 5 chooses updates; Lecture 6 makes the network a route through which both can travel.

**Handoff:** **Lecture 7 · Generalization & Regularization** asks whether a trainable network performs well beyond its training data. Do not turn this lecture into the old regularization lecture: BatchNorm may have stochastic side effects and architectural choices may affect generalization, but today’s contract is optimization and signal propagation. The clean transition is **first make the training objective reachable; next keep fitting it from becoming the whole goal**.

## Learning contract

By the end, students should be able to:

1. define and interpret the layerwise signal gauge $q_\ell=\mathbb E[(h^{(\ell)})^2]$ and gradient gauge $g_\ell=\mathbb E[(\bar h^{(\ell)})^2]$;
2. explain why local scale multipliers and activation derivatives compound with depth;
3. derive the fan-in variance rule and distinguish Xavier from He initialization;
4. calculate the five-layer ReLU prediction for Xavier and He from $q_0=g_5=1$;
5. calculate BatchNorm on $[1,2,3,4]$, including the normalization axis, affine parameters, and $\epsilon$ convention;
6. distinguish BatchNorm from LayerNorm by **which entries share statistics** and by train/eval behavior;
7. derive the residual gradient equation using column gradients and calculate the five-block plain-versus-residual comparison; and
8. diagnose a failure by moving from **symptom → measurement → one controlled test**, rather than stacking fixes.

## The persistent board model: two gauges, one network

Draw this once and keep it available for the entire lecture:

\[
h^{(\ell)}=\operatorname{ReLU}(W_\ell h^{(\ell-1)}),
\qquad \ell=1,\ldots,5,
\]

with width $4$ at every hidden layer, zero biases for the scale calculation, independent zero-mean weights, and

\[
q_\ell:=\mathbb E[(h^{(\ell)})^2],
\qquad
g_\ell:=\mathbb E[(\bar h^{(\ell)})^2],
\qquad
q_0=g_5=1.
\]

The expectation is over examples, coordinates, and—in the initialization argument—random draws. These are **raw second moments**; their square roots are RMS scales. They are not automatically variances, especially after ReLU makes the activation mean positive.

For a symmetric pre-activation, ReLU passes roughly half the squared energy. With fan-in $n=4$, the approximate per-layer second-moment multiplier is

\[
c\approx \frac12 n\operatorname{Var}(W)=2\operatorname{Var}(W).
\]

In this equal-width mean-field calculation, the backward gauge has the same approximate multiplier when read from layer $5$ toward layer $0$.

| Initialization | $\operatorname{Var}(W)$ | ReLU multiplier $c$ | $q_5=c^5q_0$ | $g_0=c^5g_5$ |
|---|---:|---:|---:|---:|
| Xavier, $2/(4+4)$ | $1/4$ | $0.5$ | $0.5^5=0.03125$ | $0.5^5=0.03125$ |
| He, $2/4$ | $1/2$ | $1$ | $1^5=1$ | $1^5=1$ |

Make students say what the table does **not** claim: an individual He-initialized network does not preserve every coordinate or every layer exactly. The table is an approximate scale prediction at initialization under simplifying assumptions.

Use this same model four ways:

1. **Activation:** the ReLU derivative is a local gate in $g_\ell$.
2. **Initialization:** Xavier gives $c=0.5$ here; He restores the lost factor of two.
3. **Normalization:** take one feature’s four-example slice to be $[1,2,3,4]$ and reset its scale explicitly.
4. **Residual path:** reinterpret the five transformations as blocks and compare a branch derivative $F'=-0.1$ with the total residual derivative $1+F'=0.9$.

## Exact 80-minute route

### ⭐ Core · exactly 55 minutes

- Opening commitment and the two gauges: 5 min.
- Depth as a product; width-4/five-layer prediction: 7 min.
- Activations as gradient gates, including saturation and inactive ReLUs: 8 min.
- Fan-in derivation, Xavier/He, and the persistent numeric: 14 min.
- Normalization formula and the exact BatchNorm calculation: 10 min.
- Residual forward/backward paths and the five-block numeric: 8 min.
- Final two-gauge synthesis and Lecture 7 handoff: 3 min.

### ⭐⭐ Should cover · exactly 15 minutes

- BatchNorm versus LayerNorm axes, CNN axes, and train/eval behavior: 9 min.
- One complete symptom → measurement → controlled-test diagnosis: 6 min.

### ⭐⭐⭐ Optional · exactly 10 minutes

- Selected notebook reveal or RMSNorm/pre-activation extension: 5 min.
- Convolutional fan-in, projection shortcuts, and modern recipe caveats: 5 min.

### Suggested clock

| Time | Route | Teaching move |
|---|---|---|
| 0–5 | ⭐ | Show the width-4, five-layer ReLU chain and make students choose Xavier normal, variance-matched Xavier uniform, He normal, or all-zero weights. Keep the answer hidden. |
| 5–12 | ⭐ | Turn a local multiplier into $c^L$; establish the width-4, five-layer board model. |
| 12–20 | ⭐ | Compare sigmoid/tanh/ReLU derivative gates; define an all-example inactive ReLU carefully. |
| 20–34 | ⭐ | Derive fan-in scaling, balance Xavier, add ReLU’s factor of two, and calculate $0.5^5$. |
| 34–44 | ⭐ | Insert normalization and work $[1,2,3,4]$ through $\mu$, $\sigma^2$, $\hat x$, $\gamma$, and $\beta$. |
| 44–52 | ⭐ | Add the residual route; derive the transpose equation and calculate $10^{-5}$ versus $0.59049$. |
| 52–61 | ⭐⭐ | Contrast BN/LN axes and their own train/eval behavior; include batched-inference nuance. |
| 61–67 | ⭐⭐ | Diagnose one failure from saved layerwise measurements and change exactly one cause. |
| 67–72 | ⭐⭐⭐ | Reveal one notebook experiment selected from the choreography below. |
| 72–77 | ⭐⭐⭐ | Choose one extension: conv fan-in, pre-activation, RMSNorm, or projection shortcuts. |
| 77–80 | ⭐ | Revisit both gauges, complete the exit ticket, and hand off to generalization. |

The labels sum to **55 core + 15 should-cover + 10 optional = 80 minutes**. If you omit optional material, spend the recovered time on predictions and student calculations rather than adding another technique.

## Teach it like this

Open on the persistent width-4, five-layer ReLU network and ask which zero-mean start should keep **both end-to-end RMS gauges** closest to one: Xavier normal with variance $1/4$, Xavier uniform with the same variance, He normal with variance $1/2$, or all-zero weights. Require a private commitment and keep the answer hidden until ReLU’s gain earns He’s factor of two. Then put two empty traces on the board: $q_0\rightarrow q_5$ forward and $g_5\rightarrow g_0$ backward. Use the scalar $0.8^{20}\approx0.0115$ only as the follow-up compounding example or in notebook 01.

Keep the causal chain tight:

1. **Depth exposes the failure.** Backprop is a product of local Jacobians; the forward computation is also a repeated product-and-gate process.
2. **Activation controls the local gate.** A saturated sigmoid has a tiny derivative; an active ReLU passes a derivative of one, while an inactive ReLU blocks that example’s path.
3. **Initialization sets the starting scale.** Random weights break symmetry, fan-in determines forward spread, fan-out matters backward, and the activation changes the gain that must be matched.
4. **Normalization recomputes a reference scale during training.** It also changes the optimizer’s parameterization; it is not adequately explained by chanting “internal covariate shift.”
5. **Residual addition creates a short path.** The arriving gradient gets an identity contribution plus a residual-Jacobian contribution.
6. **Measurement decides what to try.** Re-read $q_\ell$ and $g_\ell$ after every intervention; do not infer the failure mechanism from loss alone.

At each checkpoint, use **predict → calculate → reveal → explain**. Ask for the direction of change before asking for a decimal. The same visual grammar should survive the lecture: forward signal, backward gradient, chosen scale, and identity path should not change meaning between diagrams.

## Live numeric 1 · initialization on the width-4 ReLU stack

For one coordinate of a linear pre-activation,

\[
z=\sum_{i=1}^{n}w_ix_i.
\]

Under independent, zero-mean terms with weights independent of inputs,

\[
\operatorname{Var}(z)
=\sum_{i=1}^{n}\operatorname{Var}(w_ix_i)
=n\operatorname{Var}(w)\operatorname{Var}(x).
\]

For a linear or near-linear layer, preserving the forward variance suggests $\operatorname{Var}(W)\approx1/n_{\text{in}}$; preserving the backward variance suggests $1/n_{\text{out}}$. Xavier balances the two:

\[
\operatorname{Var}(W)=\frac{2}{n_{\text{in}}+n_{\text{out}}}.
\]

For $n_{\text{in}}=n_{\text{out}}=4$, Xavier has $\operatorname{Var}(W)=1/4$. ReLU then keeps about half the pre-activation second moment:

\[
c_{\text{Xavier+ReLU}}
\approx \frac12(4)\left(\frac14\right)=\frac12.
\]

Starting at $q_0=1$ gives

\[
q_1=0.5,\quad q_2=0.25,\quad q_3=0.125,\quad q_4=0.0625,\quad q_5=0.03125.
\]

Under the matching backward assumptions and $g_5=1$,

\[
g_4=0.5,\quad g_3=0.25,\quad g_2=0.125,\quad g_1=0.0625,\quad g_0=0.03125.
\]

He initialization restores the ReLU loss:

\[
\operatorname{Var}(W)=\frac{2}{n_{\text{in}}}=\frac12,
\qquad
c_{\text{He+ReLU}}\approx\frac12(4)\left(\frac12\right)=1.
\]

Thus the mean-field prediction is $q_5=1$ and $g_0=1$. Say **“second-moment scale is preserved approximately at initialization,”** not “the variance is exactly one forever.”

### Distribution convention

Xavier and He specify a target variance, not a unique distribution. For Xavier, these common choices are variance-matched:

\[
W_{ij}\sim\mathcal N\!\left(0,\frac{2}{n_{\text{in}}+n_{\text{out}}}\right)
\]

or

\[
W_{ij}\sim\mathcal U\!\left[-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}},
\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}\right].
\]

They have the same variance but are not identical distributions; higher moments, truncation, finite width, and framework-specific gain conventions can change realized traces. For a $k_h\times k_w$ convolution, use $n_{\text{in}}=k_hk_wC_{\text{in}}$, not just the channel count.

## Live numeric 2 · BatchNorm exactly on one feature

Treat $x=[1,2,3,4]$ as **one feature measured on four examples**, i.e. one column of a $B\times D$ activation matrix. Use the BatchNorm training-forward convention with divisor $B$:

\[
\mu=\frac{1+2+3+4}{4}=\frac52=2.5,
\]

\[
\sigma^2
=\frac{(1-2.5)^2+(2-2.5)^2+(3-2.5)^2+(4-2.5)^2}{4}
=\frac{5}{4}=1.25.
\]

With $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$,

\[
\hat x
=\frac{[-3,-1,1,3]}{\sqrt{5+4\epsilon}}.
\]

If the hand calculation explicitly takes $\epsilon=0$,

\[
\hat x
=\left[-\frac{3}{\sqrt5},-\frac{1}{\sqrt5},
\frac{1}{\sqrt5},\frac{3}{\sqrt5}\right]
\approx[-1.342,-0.447,0.447,1.342].
\]

For $\gamma=2$ and $\beta=1$,

\[
y=2\hat x+1
=\left[1-\frac{6}{\sqrt5},1-\frac{2}{\sqrt5},
1+\frac{2}{\sqrt5},1+\frac{6}{\sqrt5}\right]
\approx[-1.683,0.106,1.894,3.683].
\]

**State the convention before calculating.** The deck’s displayed decimals neglect $\epsilon$ after writing the general formula. Real implementations place a small positive $\epsilon$ inside the square root, so the normalized variance is $\sigma^2/(\sigma^2+\epsilon)$—near, but not exactly, one. Also distinguish the divisor-$B$ variance used in the training normalization from any library detail used to update a stored running-variance estimate; those bookkeeping conventions are not interchangeable.

## BatchNorm and LayerNorm: make the axes physical

For $X\in\mathbb R^{B\times D}$, draw rows as examples and columns as features.

| Question | BatchNorm (MLP) | LayerNorm |
|---|---|---|
| Which entries share $\mu,\sigma^2$? | One feature column across $B$ examples | One example row across its $D$ features |
| Parameters | One $\gamma_j,\beta_j$ per feature | Usually one $\gamma_j,\beta_j$ per normalized feature |
| Does another batch member affect this output during training? | Yes | No |
| Statistics used by the norm in training | Current batch statistics | Current example’s feature statistics |
| Statistics used by the norm in evaluation | Stored running statistics | Same per-example computation as training |
| Small-batch sensitivity | Often substantial | No batch-statistic sensitivity |
| Common home | CNNs | Transformers and sequence models |

For convolutional BatchNorm, normalize each channel over $(B,H,W)$ and keep one affine pair per channel. For a Transformer-style LayerNorm, normalization is commonly over the hidden-feature axis for each token/example position; it is not “over the entire sequence and batch.”

Be precise about inference: **BatchNorm inference can absolutely be batched.** Evaluation mode uses frozen running statistics whether the inference batch contains one example or many; it does not normalize using the current inference batch. LayerNorm’s own computation has no train/eval switch or running statistics, although the surrounding model may still contain dropout or other mode-dependent operations.

## Live numeric 3 · the residual route

For a residual block

\[
x_{\ell+1}=x_\ell+F_\ell(x_\ell),
\]

use column gradients $\bar x_\ell:=\nabla_{x_\ell}\mathcal L$ and the Jacobian $J_{F_\ell}:=\partial F_\ell/\partial x_\ell$. The chain rule gives

\[
\boxed{
\bar x_\ell
=(I+J_{F_\ell})^\top\bar x_{\ell+1}
=\bar x_{\ell+1}+J_{F_\ell}^{\top}\bar x_{\ell+1}
}.
\]

The transpose is required with column-gradient notation. If a source writes gradients as row vectors, the same chain rule appears with multiplication on the other side; choose one convention and do not mix them.

Now take the scalar branch $F(x)=-0.1x$, so $F'(x)=-0.1$:

| Stack | Local derivative | Five-block derivative | Gradient magnitude from unit upstream gradient |
|---|---:|---:|---:|
| Plain branch only | $-0.1$ | $(-0.1)^5=-0.00001$ | $10^{-5}$ |
| Residual $x+F(x)$ | $1-0.1=0.9$ | $0.9^5=0.59049$ | $0.59049$ |

The identity term creates a short additive path; it does **not** guarantee a healthy gradient. The residual Jacobian can reinforce, rotate, amplify, or partly cancel the identity contribution, multiple blocks still multiply their total Jacobians, and a projection shortcut is not an exact identity.

## Notebook demo choreography

All three notebooks are deterministic NumPy mechanism checks with retained outputs. Do not run them as a rapid slideshow. For each one, stop after the prompt, collect a prediction, calculate at least one value on the board, then reveal the trace.

### `notebooks/L06/01_signal_gradient_autopsy.ipynb`

**Cue:** immediately after defining $q_\ell$ and $g_\ell$.

1. Predict $a^{20}$ for the scalar chain with $a\in\{0.8,1.0,1.2\}$.
2. Ask which case vanishes, preserves, or explodes in both directions.
3. Run the scalar section and explain why a harmless-looking local mismatch compounds.
4. Before the random-network section, predict sigmoid + Xavier versus ReLU + He.
5. Inspect the layer where the sigmoid gradient first becomes tiny and ask whether He preserves each realization or only a useful scale range.

Use the first half in the core route. The random-network comparison can be the optional five-minute reveal.

### `notebooks/L06/02_variance_propagation.ipynb`

**Cue:** immediately after deriving $\operatorname{Var}(z)=n\operatorname{Var}(w)\operatorname{Var}(x)$.

1. Reuse the exact anchor: width $n=4$, depth $L=5$, and boundary squared gauges $q_0=g_5=1$.
2. Calculate $c=\tfrac12n\operatorname{Var}(W)$ for Xavier variance $1/4$ and He variance $2/4=1/2$, then predict $c^5$.
3. Before the independent one-layer simulation, predict post-ReLU $q_1\approx0.5$ for Xavier and $q_1\approx1$ for He. Compare the empirical averages with those predictions; fresh width-4 inputs and weights across trials estimate the expectation rather than showcase one lucky matrix.
4. Inspect the full indexed ledgers $q_\ell=c^\ell q_0$ and $g_\ell=c^{5-\ell}g_5$, including the exact endpoints $q_5=g_0=0.03125$ for Xavier and $1$ for He.
5. Ask why $0.03125$ is a **squared-gauge ratio**: the corresponding RMS ratio is $\sqrt{0.03125}\approx0.17678$, not $0.03125$. Then name which independence and half-active assumptions training will erode.

Use this notebook to calculate and empirically check the same persistent width-4 ledger carried by the lecture.

### `notebooks/L06/03_xavier_vs_he.ipynb`

**Cue:** after all four controls have been introduced.

1. Predict the **forward RMS and backward RMS separately** for sigmoid/variance-one, sigmoid/Xavier, ReLU/Xavier, and ReLU/He before revealing the 24-layer traces. Do not force one total ranking: the two directions can disagree. Choose the configuration expected to be healthiest on both and justify it using weight gain and activation derivative.
2. On the $B\times D$ array, point to the entries that share statistics under BN, then under LN; ask what changes when a fifth example joins the batch.
3. Calculate $(-0.1)^5$ and $0.9^5$ before running the residual section.
4. End by having students complete: activation controls the local gate; initialization controls starting scale; normalization controls scale over a chosen axis during training; residual addition changes $J_F$ to $I+J_F$.

If class time is tight, assign this third notebook as the post-class synthesis because it joins all four controls.

## Diagnose: symptom → measurement → controlled test

Save an initialization-time forward/backward pass before changing the model. Log activation mean and RMS, pre-activation range, per-unit ReLU inactivity, gradient RMS or norm by depth, gradient-to-weight ratio, update-to-weight ratio, and finite-value checks. There is no universal healthy gradient-to-weight threshold; the useful evidence is an abrupt layerwise change or a reproducible trend.

| Symptom | Measurement next | One controlled test |
|---|---|---|
| Activation RMS shrinks layer by layer before any update | $q_\ell$ before and after each activation | With seed, data, depth, and activation fixed, replace only Xavier by He for the ReLU layers. |
| Signal explodes or becomes non-finite | First layer where activation/pre-activation RMS jumps; weight scale | Before training, rerun with only the initialization gain reduced. If failure begins only after updates, test the learning rate separately. |
| Many ReLUs output zero | Fraction zero per example **and** units negative for every measured example | Keep weights/data fixed and replace only the gate with a small-slope leaky ReLU for one diagnostic run. |
| Early-layer gradients are tiny while late gradients are finite | $g_\ell$ or gradient RMS by depth on the same batch | Add a matched identity shortcut to one stack while preserving branch weights and compare the trace. |
| Training works shallow but fails when depth increases | Both gauges at matched initialization and width | Hold seed, width, data, optimizer, and update budget fixed; change only depth. |
| BN model differs sharply between training and evaluation | Current-batch statistics, stored running statistics, and module mode | On a cloned/frozen model and the same inputs, compare batch-stat and running-stat outputs without an optimizer update. |
| Tiny batches make training noisy | Variation of per-channel BN mean/variance across batches | Replace only BN with a batch-independent norm appropriate to the architecture and repeat the same-seed run. |

The test identifies a mechanism; it is not automatically the final production recipe. Once evidence changes, re-measure both gauges before making a second change.

## Heads-up for the instructor

- **RMS/second moment is not variance.** $q_\ell=\mathbb E[h^2]$ and $g_\ell=\mathbb E[\bar h^2]$ are raw second moments; RMS is their square root. Variance subtracts the squared mean. ReLU outputs are generally positive-mean, so calling $q_\ell$ “the variance” is wrong.
- **The derivations are mean-field approximations.** Independence, zero-mean weights, weak correlations, symmetric pre-activations, comparable coordinates, and activation-gate independence are most plausible at initialization and finite width only approximates them. Training builds correlations and changes the distributions.
- **A scalar gauge can hide bad directions.** Average RMS may look healthy while a Jacobian has a few exploding or collapsing singular directions. The gauge is a useful first diagnostic, not a proof of dynamical isometry.
- **Randomness breaks symmetry; scale controls propagation.** All-zero weights give identical hidden units identical gradients. Zero biases are usually fine because random incoming weights already distinguish units.
- **Xavier’s Gaussian and uniform forms are variance-matched alternatives.** Do not call them the same distribution. Xavier is a compromise between forward fan-in and backward fan-out, best motivated for linear/near-linear or tanh-like layers; it does not cure saturation.
- **He matches ReLU’s approximate gain.** The factor two comes from $\mathbb E[\operatorname{ReLU}(z)^2]\approx\tfrac12\mathbb E[z^2]$ for symmetric $z$, not from a universal law that half of every finite batch is active.
- **Initialization governs the start, not the whole run.** Normalization can reduce sensitivity, but neither normalization nor residuals makes initialization irrelevant. The learning rate and optimizer from Lecture 5 can move the model out of its initial regime immediately.
- **“Dead ReLU” needs a data qualifier.** If a unit is negative for every current example, its own incoming parameters receive zero gradient from those examples. A changing upstream representation can later move its pre-activation positive, so “a ReLU can never revive” is too absolute; for a fixed input representation and fixed incoming parameters, it cannot revive itself through that zero local derivative.
- **BatchNorm is more than an internal-covariate-shift story.** It explicitly controls scale and changes the parameterization/optimization geometry; batch noise may also matter. The original ICS motivation is historically important, but it is not a sufficient settled explanation of why BN helps.
- **BatchNorm’s clocks and estimators must be named.** Training normalizes with current-batch statistics and updates running state; evaluation normalizes with frozen running state. Framework conventions for momentum and stored variance can differ. `model.eval()` does not mean inference must be unbatched.
- **LayerNorm’s own operation is mode-independent.** It uses each example/token’s normalized features in both modes and has no running statistics. Do not generalize this statement to the whole model if dropout is present.
- **Residual means path, not guarantee.** $I+J_F$ can still have small or large singular values, and a projection shortcut changes the direct term to the projection Jacobian. Branch scaling, normalization placement, activation placement, and learning rate remain coupled choices.
- **Pre-activation has a precise advantage.** Keeping norm and activation inside $F$ can leave the addition’s identity route ungated. Post-activation places another derivative after the addition. This is an architectural tendency, not proof that every pre-norm or pre-activation model is stable.
- **Modern recipes are starting points, not laws.** ReLU/He is a strong MLP/CNN baseline; CNNs often pair He-like init, BN, and residual blocks; Transformers often use LayerNorm or RMSNorm, residual streams, GELU/SiLU, warmup, and architecture-specific residual scaling. GELU/SiLU gains, pre/post-norm choice, depth scaling, and parameterization differ across model families. Fixup-style results also show that carefully scaled residual networks can train without normalization.

## Where students stumble—and the fix

- **“$q_\ell$ is the activation variance.”** Ask whether ReLU output has mean zero. Rewrite $\operatorname{Var}(h)=\mathbb E[h^2]-\mathbb E[h]^2$ and label the gauge “second moment / RMS.”
- **“A $0.5$ multiplier leaves half the signal after five layers.”** Make them multiply: $0.5^5=0.03125$. Depth compounds; it does not subtract.
- **“Xavier preserves ReLU networks.”** Return to the width-4 row: Xavier preserves the linear pre-activation scale, then ReLU loses roughly half the squared energy. He restores that factor.
- **“He makes every layer exactly unit variance.”** Contrast a distributional initialization prediction with one finite realization, then inspect the notebook trace.
- **“Sigmoid has a derivative, so it passes gradients.”** Calculate $\sigma'(5)\approx0.00665$ and ask what five such gates multiply to.
- **“Every zero ReLU output is a dead unit.”** One zero on one example is normal. A useful diagnostic is whether the same unit is inactive for every relevant example.
- **“A dead ReLU can never return.”** Separate its zero local parameter gradient from changes in the representation feeding it. Earlier layers can move its input distribution.
- **“BatchNorm normalizes each example.”** Point down a feature column of the $B\times D$ grid. LayerNorm points across a row.
- **“BatchNorm cannot serve a batch at inference.”** It can serve any inference batch size; evaluation uses stored running statistics rather than that batch’s statistics.
- **“BN works because it eliminates internal covariate shift.”** Replace the slogan with what is directly visible: controlled scale and a changed, often easier-to-optimize parameterization.
- **“Residual means the gradient is one.”** The local Jacobian is $I+J_F$, not just $I$. In the numeric it is $0.9$, and across five blocks it is $0.59049$.
- **“The deeper plain model’s higher error is overfitting.”** If its **training** error is higher, that is an optimization degradation problem. Generalization error belongs to Lecture 7.
- **“Use activation + He + BN + residuals everywhere.”** Ask which measured failure each proposed change targets. Architecture-specific recipes and controlled comparisons beat ritual stacking.

## If a student asks…

- **“Why use RMS instead of standard deviation?”** RMS is $\sqrt{\mathbb E[h^2]}$ and tracks total squared signal energy even when the activation mean is nonzero. Standard deviation removes the mean; after ReLU that can hide part of the propagated scale.
- **“Why does backward scaling involve fan-out?”** Each input receives a sum of contributions from the units it feeds. Under the same independence argument, the number of outgoing terms controls the backward sum’s second moment.
- **“Why not satisfy fan-in and fan-out exactly?”** If they differ, one scalar weight variance cannot equal both $1/n_{\text{in}}$ and $1/n_{\text{out}}$. Xavier uses a symmetric compromise; other parameterizations may prioritize one direction.
- **“Is Xavier uniform better than Xavier normal?”** Neither follows from the variance derivation. They share the target variance but differ in higher moments and tails; use the architecture/framework convention and verify actual traces.
- **“Should GELU always use He initialization?”** There is no single universal gain for every smooth gate and architecture. Start from the model family’s validated initialization and residual-scaling recipe, then measure both gauges.
- **“If He works, why normalize?”** He targets the starting distribution under approximations. Normalization recomputes a reference scale as parameters and representations change and also alters the optimization parameterization.
- **“Does BatchNorm require one example at inference?”** No. It supports one or many. Evaluation mode uses stored running statistics for each example; the current inference batch does not define the normalization.
- **“Why do Transformers prefer LayerNorm or RMSNorm?”** Their normalization is batch-independent and naturally applies along the hidden-feature axis for each token. This avoids coupling predictions to other batch members and avoids running-statistic bookkeeping.
- **“Is BatchNorm a regularizer?”** Its batch dependence can add noise and influence generalization, but its role here is trainability through scale control and reparameterization. Treat any regularization effect as recipe-dependent and defer the generalization question to Lecture 7.
- **“Do residuals remove vanishing gradients?”** They provide shorter additive routes. They do not force $I+J_F$ to preserve every direction, and poor branch scale, norm placement, or learning rate can still destabilize the stack.
- **“What if the block changes width?”** Replace the identity with a projection $P$: $x_{\ell+1}=P_\ell x_\ell+F_\ell(x_\ell)$. The direct backward term is then $P_\ell^\top\bar x_{\ell+1}$, so it is no longer an untouched identity path.
- **“Can a very deep residual network train without normalization?”** Yes, with carefully designed branch scaling and initialization, as Fixup demonstrates. That is evidence that initialization, normalization, and residual paths are coupled controls—not that normalization never helps.

## Exit ticket

Give students these three prompts before the closing line:

1. With width $4$, ReLU, and Xavier variance $1/4$, what is $q_5$ from $q_0=1$? Why does He change the answer?
2. In a $B\times D$ table, which axis does BN reduce and which axis does LN reduce? What changes at evaluation?
3. With column gradients, where does the transpose appear in the residual backward equation, and why is the identity term a path rather than a guarantee?

Expected compression: **activation chooses the local gate; initialization chooses the starting multiplier; normalization resets scale over a declared axis; residual addition supplies a short identity contribution.**

## If you are short on time

Cut the activation-zoo comparison, convolutional fan-in calculation, RMSNorm details, projection shortcuts, pre- versus post-activation comparison, historical ICS discussion, and in-class notebook execution. Assign all three notebooks as predict-before-run follow-along work.

Never cut:

1. both gauges and the distinction between second moment, RMS, and variance;
2. the width-4, five-layer calculation $0.5^5=0.03125$ versus $1^5=1$;
3. the exact $[1,2,3,4]$ BatchNorm calculation and the $\epsilon$ convention;
4. BN versus LN axes plus BatchNorm train/eval behavior;
5. the column-gradient residual equation and $10^{-5}$ versus $0.59049$;
6. one symptom → measurement → controlled test; and
7. the handoff from trainability to Lecture 7 generalization.

### 45-minute emergency route

| Time | Teaching move |
|---|---|
| 0–4 | Opening commitment and define $q_\ell,g_\ell$. |
| 4–10 | Depth compounding on the persistent network. |
| 10–16 | Activation gates: one sigmoid numeric and the careful inactive-ReLU definition. |
| 16–26 | Fan-in derivation, Xavier/He, and $0.5^5$ versus $1^5$. |
| 26–34 | BatchNorm calculation, BN/LN axes, and train/eval distinction. |
| 34–41 | Residual transpose equation and five-block numeric. |
| 41–43 | One diagnosis from the table. |
| 43–45 | Exit ticket, synthesis, and Lecture 7 handoff. |

## Closing line

“Backprop supplies the gradient and the optimizer chooses the update; activation, initialization, normalization, and residual paths make sure useful signals and gradients can survive the depth. Now that the network can reach a low training loss, Lecture 7 asks whether what it learned will generalize.”

## Primary references

- Glorot & Bengio (2010), [*Understanding the Difficulty of Training Deep Feedforward Neural Networks*](https://proceedings.mlr.press/v9/glorot10a.html).
- He et al. (2015), [*Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*](https://arxiv.org/abs/1502.01852).
- Ioffe & Szegedy (2015), [*Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*](https://arxiv.org/abs/1502.03167).
- Santurkar et al. (2018), [*How Does Batch Normalization Help Optimization?*](https://arxiv.org/abs/1805.11604).
- Ba, Kiros & Hinton (2016), [*Layer Normalization*](https://arxiv.org/abs/1607.06450).
- He et al. (2016), [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385).
- He et al. (2016), [*Identity Mappings in Deep Residual Networks*](https://arxiv.org/abs/1603.05027).
- Zhang et al. (2019), [*Fixup Initialization: Residual Learning Without Normalization*](https://openreview.net/forum?id=H1gsz30cKX).

Treat the derivations as initialization-scale arguments and the deterministic notebooks as mechanism checks, not universal guarantees or benchmark evidence.
