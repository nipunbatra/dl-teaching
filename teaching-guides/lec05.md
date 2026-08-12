# Lecture 5 · Optimization for Deep Learning · Teaching Guide
*First-time-instructor companion. Audience: undergraduates who have completed the public Lectures 1–4.*

## The spine (say this in two sentences)

Backprop gives a minibatch gradient, but a gradient is only a measurement: an optimizer must decide what history to remember, how that history rescales each coordinate, and what scheduled step to apply. Follow one dense/rare gradient stream from SGD to momentum, AdaGrad, RMSProp, Adam, and AdamW so every new formula is earned as a specific answer to “what should the past change about this step?”

## Where it sits

**Prerequisites in the public sequence:**

- **Lecture 1 · Where Deep Learning Losses Come From:** a scalar loss is the training objective whose sensitivity we measure.
- **Lecture 2 · From Linear Models to Neural Networks:** parameters live inside a composed model, so one update changes many predictions.
- **Lecture 3 · Calculus Toolkit** (`L3A.pdf`): gradients, directional derivatives, and the first-order Taylor model. The optional steepest-descent derivation uses these directly.
- **Lecture 4 · Computation Graphs & Backpropagation** (`L3.pdf`): backprop supplies $g_t=\nabla_\theta \mathcal L_{\mathcal B_t}(\theta_t)$. This lecture begins where that computation ends.

Do **not** assume momentum or an LR finder was taught in an earlier numbered lecture; in the current public sequence, momentum is developed here.

**Handoff:** the immediate next lecture is **Lecture 6 · Making Deep Networks Trainable**—initialization, activation choice, normalization, and residual paths make useful gradients and activations survive depth. Lecture 7 then revisits AdamW’s decoupled weight decay as one part of generalization and regularization. Every later model-training lecture relies on today’s distinction between the measured gradient, optimizer state, and the learning-rate clock.

## Learning contract

By the end, students should be able to:

1. distinguish $g_t$, optimizer state, and the applied update $\Delta\theta_t$;
2. explain momentum as direction memory and AdaGrad/RMSProp as coordinate-wise scale memory;
3. calculate the first few states and updates of AdaGrad, RMSProp, and Adam;
4. derive Adam’s bias-correction denominator from missing EMA mass;
5. explain precisely why coupled L2 and decoupled AdamW behave differently under adaptive preconditioning;
6. compute a warmup–cosine clock using an explicit indexing convention; and
7. diagnose an optimizer symptom by measuring the relevant mechanism before changing algorithms.

## 80-minute route

### ⭐ Core · about 55 minutes

- Opening commitment and the dense/rare stream: 5 min.
- Gradient versus update; SGD on the stream; quadratic prediction: 8 min.
- Minibatch noise and momentum as direction memory: 9 min.
- AdaGrad’s activity ledger and its long-run limitation: 10 min.
- RMSProp’s fading scale memory and cold start: 7 min.
- Adam as two memories plus bias correction: 10 min.
- AdamW’s decoupled shrinkage calculation: 6 min.

The core is a **chain of mechanisms**, not an optimizer catalogue. Preserve the recurring stream and the predict-before-reveal moments even if other material is cut.

### ⭐⭐ Should cover · about 15 minutes

- Warmup and cosine as a clock multiplying the optimizer’s update scale.
- The exact $t=0,4,9,55,100$ schedule calculation.
- Scheduler advancement at the optimizer-step boundary, especially under gradient accumulation.
- The final three-question synthesis: direction, scale, clock.

### ⭐⭐⭐ Optional · up to 10 minutes

- First-order constrained steepest-descent derivation.
- Full ravine geometry and the four method-specific trajectories.
- Starting-recipe and fair-comparison tables.
- The detailed symptom table, global-norm clipping calculation, and loss-spike investigation.

If using optional material, take the time from the ravine discussion or notebook demo—not from the opening commitment, dense/rare calculation, AdamW comparison, or final synthesis.

### Suggested clock

| Time | Teaching move |
|---|---|
| 0–5 | Students commit on the dense/rare question; keep answers hidden. |
| 5–13 | Separate backprop’s measurement from SGD’s update; predict the quadratic. |
| 13–22 | Make minibatch noise concrete; build momentum as direction memory. |
| 22–39 | Build AdaGrad’s all-history ledger, then replace it with RMSProp’s fading ledger. |
| 39–52 | Combine the two memories into Adam; derive and calculate bias correction. |
| 52–60 | Do the coupled-L2 versus AdamW calculation. |
| 60–71 | Compute warmup–cosine and establish the optimizer-update clock. |
| 71–76 | Diagnose one loss spike without switching optimizers reflexively. |
| 76–80 | Revisit the opening answer, complete the exit ticket, and hand off to L6. |

## Teach it like this

Open on the table $g_1=(4,0), g_2=(4,0), g_3=(4,0.4)$. Ask which stored information could reveal that the second coordinate is *rare* rather than merely small, and make every student commit before discussion. Do not reveal “Adam” as the answer: the answer is **past squared gradients per coordinate**, the information an adaptive optimizer needs.

Keep the same two-coordinate stream visible as the lecture’s evolving visual anchor. Use the deck’s colour grammar consistently: blue is the measured gradient, teal is direction memory, green is scale memory, amber is the applied update or clock, and red marks warnings or explicitly labelled target landmarks. Every transition should sound like a response to an observed limitation:

1. **SGD:** one global rate preserves the raw 10× scale gap and remembers nothing.
2. **Momentum:** average direction so alternating noise cancels and persistent direction accumulates; rarity is still invisible.
3. **AdaGrad:** store squared activity per coordinate so a rarely active coordinate retains leverage; permanent memory can eventually make active coordinates move too slowly.
4. **RMSProp:** replace the permanent sum with an EMA so old scale evidence fades; now confront the zero-start missing mass.
5. **Adam:** put direction memory in the numerator and scale memory in the denominator, then correct both EMAs for their zero initialization.
6. **AdamW:** keep proportional parameter shrinkage outside the adaptive data-gradient preconditioner.
7. **Schedule:** multiply the resulting update by an explicit time-dependent scale; make clear which event advances its clock.

At each `[Q]`, require a private prediction or neighbour discussion before revealing `[A]`. At each `[D]`, calculate at least the first line aloud and make students supply the next. Use the diagrams to carry the causal story—measurement → remembered direction/scale → scheduled update—rather than decorating a formula tour.

Use `01_gd_quadratic.ipynb` twice as the deck intends: first complete the fixed-rate section after the quadratic, then return after schedules are introduced to predict the time-varying multiplier $1-2\eta_t$. Use `06_optimizer_comparison_pytorch.ipynb` only after students have calculated state by hand; the notebook verifies a mechanism, not a leaderboard.

## Live numeric 1 · one dense coordinate, one rare coordinate

Write this at the start and leave it on the board:

\[
\theta_0=(1,1),\qquad
g_1=(4,0),\quad g_2=(4,0),\quad g_3=(4,0.4).
\]

The first coordinate $d$ is dense; the second $r$ appears only at step 3 and is 10× smaller when it appears.

### SGD, $\eta=0.1$

\[
\Delta\theta_1=(-0.4,0),\quad
\Delta\theta_2=(-0.4,0),\quad
\Delta\theta_3=(-0.4,-0.04).
\]

Thus

\[
\theta_1=(0.6,1),\quad
\theta_2=(0.2,1),\quad
\theta_3=(-0.2,0.96).
\]

Point: one global $\eta$ preserves the 10× magnitude gap at step 3.

### Momentum direction memory, $\beta=0.9$

For the separate one-coordinate sequence $g=(10,8,11)$, using the deck’s normalized-EMA convention

\[
m_t=0.9m_{t-1}+0.1g_t,\qquad m_0=0,
\]

the exact states are $m_1=1.00, m_2=1.70, m_3=2.63$. Use this to establish smoothing and cold-start missing mass; do not yet call it Adam bias correction.

### AdaGrad ledger

Elementwise $G_t=G_{t-1}+g_t^2$ gives:

| $t$ | $G_t$ | $g_t/(\sqrt{G_t}+\epsilon)$ |
|---|---|---|
| 1 | $(16,0)$ | $(1,0)$ |
| 2 | $(32,0)$ | $(0.707,0)$ |
| 3 | $(48,0.16)$ | $(0.577,1)$ |

With $\eta=0.1$, the dense updates are $-0.100,-0.071,-0.058$, the rare updates are $0,0,-0.100$, and the final displayed state is approximately $\theta_3=(0.772,0.900)$. The rare coordinate’s first nonzero normalized magnitude is 1 even though its raw gradient is 10× smaller.

### RMSProp fading ledger, $\rho=0.9$

With $v_0=(0,0)$ and $v_t=0.9v_{t-1}+0.1g_t^2$:

| $t$ | $v_t$ | $g_t/(\sqrt{v_t}+\epsilon)$ |
|---|---|---|
| 1 | $(1.600,0)$ | $(3.162,0)$ |
| 2 | $(3.040,0)$ | $(2.294,0)$ |
| 3 | $(4.336,0.016)$ | $(1.921,3.162)$ |

At a coordinate’s first nonzero observation, $g_1/\sqrt{v_1}=\operatorname{sign}(g_1)/\sqrt{1-\rho}$, hence magnitude $3.16$ for $\rho=0.9$. This is a zero-start effect, not evidence that the first sample deserves extra trust.

### Adam at step 3, $\beta_1=0.9, \beta_2=0.999$

For the recurring stream, students should obtain:

| Quantity at $t=3$ | dense $d$ | rare $r$ |
|---|---:|---:|
| $m_3$ | 1.084 | 0.040 |
| $\hat m_3$ | 4.000 | 0.148 |
| $v_3$ | 0.047952 | 0.000160 |
| $\hat v_3$ | 16.000 | 0.05339 |
| $\hat m_3/\sqrt{\hat v_3}$ | 1.000 | 0.639 |

The scale memory lifts the rare coordinate relative to its raw 0.1 ratio; direction memory tempers it because the first two batches supplied zero evidence in that coordinate.

Before this table, do the first-step checkpoint exactly: for $g_1=(4,0.4)$, $m_1=(0.4,0.04)$, $v_1=(0.016,0.00016)$, $\hat m_1=(4,0.4)$, $\hat v_1=(16,0.16)$, and—ignoring $\epsilon$—$\hat m_1/\sqrt{\hat v_1}=(1,1)$.

## Live numeric 2 · coupled L2 versus AdamW

Hold the preconditioner fixed to isolate the geometry:

\[
\theta=(1,1),\qquad \hat v=(16,0.16),\qquad \eta=0.01,\qquad \lambda=0.1.
\]

If an L2 contribution $\lambda\theta$ is passed through the adaptive denominator, its isolated approximate update contribution is

\[
-\eta\lambda\theta/\sqrt{\hat v}.
\]

Therefore:

| Coordinate | Coupled L2 contribution | Decoupled AdamW contribution |
|---|---:|---:|
| dense | $-0.01(0.1)/4=-0.00025$ | $-\eta\lambda\theta_d=-0.001$ |
| rare | $-0.01(0.1)/0.4=-0.0025$ | $-\eta\lambda\theta_r=-0.001$ |

Make students say the result: with equal parameter values, coupled L2 shrinks the rare coordinate 10× more in this fixed-$\hat v$ comparison, whereas AdamW applies the same multiplicative shrink factor $1-\eta\lambda$ to both. Then state the caveat: in a truly coupled adaptive optimizer, adding $\lambda\theta$ also changes the moment estimates; holding $\hat v$ fixed is an explanatory isolation, not a simulated full trajectory.

## Supporting numerics worth preserving

- **Quadratic prediction:** for $\mathcal L(\theta)=(\theta-3)^2$, $e_{t+1}=(1-2\eta)e_t$. Alternating error that shrinks by half requires $1-2\eta=-0.5$, so $\eta=0.75$.
- **Warmup–cosine clock:** with $\eta_{\max}=0.1, \eta_{\min}=0, T_w=10, T=100$, the deck’s convention gives $\eta_0=0.01, \eta_4=0.05, \eta_9=0.10, \eta_{55}=0.05, \eta_{100}=0$. Here $t=100$ is the unexecuted endpoint; a 100-update run applies $\eta_0,\ldots,\eta_{99}$.
- **Global-norm clipping, optional:** $g=(6,8)$ has norm 10; with threshold $c=5$, multiply by $5/10$ to obtain $(3,4)$, norm 5.

## Heads-up for the instructor

- **A gradient is not an update.** Backprop measures $g_t$; the optimizer uses $g_t$, stored state, and the schedule to construct $\Delta\theta_t$. Keep the sign convention $\theta_{t+1}=\theta_t+\Delta\theta_t$ visible.
- **The minibatch gradient is conditionally unbiased under the stated sampling scheme.** $\mathbb E[g_t\mid\theta_t]=\nabla\mathcal L(\theta_t)$ does not mean every minibatch points downhill or lowers the full training loss.
- **State the momentum convention.** The deck uses a normalized EMA, $m_t=\beta m_{t-1}+(1-\beta)g_t$. Some libraries and texts use an unnormalized velocity; their numerical states and effective learning rates differ. In the ravine comparison, normalized $\eta=0.06$ is implemented by the equivalent unnormalized-buffer rate $0.006$ when $\beta=0.9$.
- **Squares and divisions are elementwise.** $g_t^2$, $\sqrt{v_t}$, and the adaptive division are coordinate-wise. Adam’s $v_t$ estimates a raw second moment, not the variance $\mathbb E[(g-\mathbb E g)^2]$.
- **AdaGrad does not literally reach zero in a finite ordinary step.** Its effective rate on persistently active coordinates tends toward zero as its accumulator grows, which can make long training runs practically freeze.
- **$\epsilon$ is part of the algorithm.** It prevents division by zero and can materially affect extremely low-scale coordinates; it is not a proof device or a general stabilizer.
- **Bias correction repairs zero-initialized EMA mass.** For constant $g$, $m_t=(1-\beta_1^t)g$ and $v_t=(1-\beta_2^t)g^2$, so division by the missing mass restores $g$ and $g^2$ exactly. In a noisy, changing problem, this does **not** make a minibatch gradient statistically unbiased, certify that a moment estimate is accurate, or guarantee descent.
- **Do not teach “Adam’s first step is 10× too small.”** The raw first moment alone is 0.1 of $g$ when $\beta_1=0.9$, but Adam’s update is a ratio and the second moment is also biased. With $\beta_1=0.9, \beta_2=0.999$, the uncorrected ratio is scaled by $0.1/\sqrt{0.001}\approx3.16$, while the correctly bias-adjusted first-step ratio is approximately $\operatorname{sign}(g)$, ignoring $\epsilon$.
- **Correction factors are not update multipliers.** At $t=1$, the raw-moment correction factors are 10 for $m$ and 1000 for $v$; because the update uses $\hat m/\sqrt{\hat v}$, it is wrong to describe the Adam step as 10× or 1000× larger.
- **Warmup is a schedule, not a repair for bias correction.** The deck’s defensible claim is operational: warmup deliberately limits the initial applied learning-rate scale and is useful in many established large-model recipes. It is not universally required, does not make $\hat v_t$ “trustworthy,” and cannot rescue an unstable peak rate, bad data, exploding activations, or a broken model.
- **AdamW means decoupled decay, not “better Adam” by theorem.** The decay term applies proportional shrinkage outside the adaptive data-gradient update. Biases and normalization gains are often excluded, but exclusions are recipe- and model-dependent.
- **Optimizer comparisons require method-specific rates.** The ravine panels deliberately use different learning rates; they illustrate mechanisms, not rankings. Compare at equal update budget and record optimizer, rate, betas, $\epsilon$, decay, schedule, batch size, and seeds.
- **The schedule’s clock must match its definition.** An update-based schedule advances once per `optimizer.step()`, not once per microbatch. State the indexing convention to avoid a silent off-by-one.
- **Clipping is an intervention, not a diagnosis.** Global-norm clipping preserves the gradient direction while capping magnitude. It can contain one dangerous step but does not make a chronically unstable rate or architecture correct.

## Where students stumble—and the fix

- **“The gradient is the step.”** Draw the pipeline again and ask what changes if two optimizers receive the same $g_t$. Same measurement, different state, different update.
- **“Unbiased means each minibatch descends.”** Contrast one noisy draw with an average over repeated draws at the same $\theta_t$.
- **“Momentum gives every parameter its own learning rate.”** It stores direction history; the rare-coordinate example shows that it does not by itself identify coordinate-wise activity scale.
- **“AdaGrad failed, so squared-gradient history was the wrong idea.”** The useful idea is the coordinate-wise ledger; the failure is that old evidence never fades. RMSProp changes the memory, not the underlying idea.
- **“Bias correction makes Adam unbiased.”** Ask: unbiased with respect to what? It corrects missing EMA mass from zero initialization; it does not turn a noisy minibatch into the population gradient.
- **“Warmup is needed because Adam’s $v_t$ starts at zero.”** Bias correction already handles the deterministic missing mass. Warmup controls the applied early scale and must be justified as part of the full model/training recipe.
- **“L2 and AdamW are the same because both use `weight_decay`.”** Re-run the fixed-$\hat v$ calculation: coupled L2 is preconditioned; decoupled decay is applied outside that preconditioner.
- **“Adam is always better than SGD.”** There is no universal winner. Use a validated regime-specific baseline, tune the rate/schedule fairly, and compare held-out metrics at equal update budget.
- **“Step the scheduler after every batch object.”** Under gradient accumulation, several microbatches can produce one optimizer update. Advance an update-based scheduler only at that boundary.
- **“A spike means switch optimizers.”** Preserve evidence: rerun the batch and inspect its data, gradient norm, activation scale, learning rate, and optimizer state before changing one suspected cause.

## If a student asks…

- **“Why can AdaGrad give the rare coordinate a normalized magnitude of 1?”** At its first nonzero appearance, that coordinate has $G=g^2$, hence $g/\sqrt{G}=\operatorname{sign}(g)$, ignoring $\epsilon$. Its small raw magnitude is interpreted relative to its own sparse history, not compared directly with the dense coordinate.
- **“Is Adam just sign descent?”** Only the bias-corrected first nonzero update from zero state reduces approximately to the sign, ignoring $\epsilon$. Later, $\hat m_t$ and $\hat v_t$ summarize different histories, so both magnitude and direction depend on the stream.
- **“Why is $v_t$ called the second moment if it is not a variance?”** The second raw moment is $\mathbb E[g^2]$; variance subtracts the squared mean. Adam tracks an EMA of squared gradients, so “second raw moment” is the precise phrase.
- **“Does bias correction become irrelevant immediately?”** No. The missing mass $1-\beta^t$ approaches one at a rate set by $\beta$, and $\beta_2=0.999$ evolves much more slowly than $\beta_1=0.9$. But reason about the combined ratio, not either correction factor in isolation.
- **“Then why use warmup at all?”** It is a controlled way to begin with a smaller global update scale and ramp to the peak prescribed by a model’s training recipe. Its value is empirical and system-dependent; verify it alongside the peak rate, normalization, batch size, and total update budget.
- **“Why does AdamW apply the same shrink if two weights differ?”** It applies the same *multiplicative factor* $1-\eta\lambda$, not the same absolute decrement. The displayed decrements match only because both example parameters equal 1.
- **“Should biases and normalization parameters receive weight decay?”** Often they are excluded because their roles and scales differ from ordinary weights, but this is a recipe choice—not part of AdamW’s mathematical definition. Follow and validate the architecture’s established parameter groups.
- **“Which learning rate should I use?”** Start from a published, validated recipe for the model regime, then run a controlled rate/schedule sweep. `3e-4` is not a universal AdamW constant, and SGD and adaptive methods generally need different scales.
- **“Should `scheduler.step()` come before or after `optimizer.step()`?”** Follow the scheduler API and define what $t$ means. For the update-counted convention in this deck and common PyTorch schedulers, advance it at the optimizer-update boundary, after the parameter update, and never once per accumulated microbatch.
- **“Can clipping fix exploding gradients?”** It can cap a particular update and buy diagnostic time. If norms are chronically large, investigate rate, data, normalization, activations, and architecture; clipping alone hides the cause.

## If you are short on time

Cut the constrained steepest-descent derivation, the four-panel ravine comparison, the regime recipe table, and the detailed clipping/loss-spike material. Demonstrate only the fixed-rate portion of `01_gd_quadratic.ipynb`; assign the schedule portion and `06_optimizer_comparison_pytorch.ipynb` as follow-along work.

Never cut:

1. the opening dense/rare commitment and its final revisit;
2. the mechanism chain **SGD → momentum → AdaGrad → RMSProp → Adam**;
3. the first-step bias-correction calculation;
4. the exact coupled-L2 versus AdamW calculation;
5. one warmup–cosine clock calculation plus the optimizer-step boundary; and
6. the final compression into **direction, scale, clock**.

In a 45-minute emergency route: opening/SGD 6 min; momentum 6 min; AdaGrad/RMSProp 10 min; Adam and correction 10 min; AdamW 6 min; schedule boundary 4 min; revisit and handoff 3 min.

## Closing line

“Backprop measures the gradient; the optimizer remembers direction and scale; the schedule decides how strongly to act. Adam gives every coordinate a history-dependent effective step, AdamW keeps parameter shrinkage outside that adaptive geometry, and next we make those gradients and activations survive a deep network.”

## Primary references

- Duchi, Hazan & Singer (2011), [*Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*](https://jmlr.org/papers/v12/duchi11a.html).
- Tieleman & Hinton (2012), *Lecture 6.5—RMSProp*, Neural Networks for Machine Learning.
- Kingma & Ba (2015), [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980).
- Loshchilov & Hutter (2019), [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101).
- Vaswani et al. (2017), [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)—a prominent warmup-and-decay recipe, not evidence that warmup is universally necessary.

All optimizer trajectories in the deck are generated from the displayed loss and update rules. Treat the deterministic notebooks as mechanism checks, not benchmark evidence.
