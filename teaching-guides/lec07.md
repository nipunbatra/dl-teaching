# Lecture 7 · Generalization & Regularization · Teaching Guide
*First-time-instructor companion. Audience: undergraduates who have completed the public Lectures 1–6 and one introductory ML course.*

## The spine (say this in two sentences)

Lecture 6 made a deep network trainable: useful signals and gradients can now survive long enough for the optimizer to fit the training set. Lecture 7 changes the success criterion—read training and validation together, diagnose the failure before prescribing a remedy, and accept early stopping, weight decay, dropout, or data/target perturbations only when held-out evidence improves.

The line to repeat is:

> **First make the model fit. Then regularize the specific failure you observe. Validation is the judge; test reports once.**

## Where it sits

**Builds on the public sequence:**

- **Lecture 1 · Where Deep Learning Losses Come From:** empirical risk, likelihood-derived losses, Gaussian priors, MAP, and $L_2$ penalties; this deck will state its factor-$1/2$ convention explicitly.
- **Lecture 2 · From Linear Models to Neural Networks:** capacity, nonlinear decision boundaries, and hidden activations.
- **Lecture 4 · Computation Graphs & Backpropagation:** gradients say how each parameter changes the loss.
- **Lecture 5 · Optimization for Deep Learning:** SGD and AdamW turn gradients into updates; AdamW already introduced decoupled decay.
- **Lecture 6 · Making Deep Networks Trainable:** activation, initialization, normalization, and residual paths make the training objective reachable.

Open with the exact Lecture 6 handoff: **a network can now optimize successfully, but low training loss answers only whether it fits examples it has already seen.** What stays fixed is the supervised model, loss, backpropagation, and optimizer loop. What changes is the question: from “can it fit?” to “which fitting solution survives unseen data?”

**Handoff:** **Lecture 8 · Convolutional Neural Networks** asks whether the architecture itself can encode a defensible invariance. Augmentation in this lecture says “the label should survive this transformation”; convolution next lecture builds locality and translation equivariance into the prediction rule. Close with: **today we taught invariances through data; next we put one into the architecture.**

## Learning contract

By the end, students should be able to:

1. distinguish empirical training loss, population risk, validation estimates, and a final test estimate;
2. state the three split roles: train updates parameters, validation selects decisions, test reports once;
3. diagnose underfitting versus overfitting from both absolute performance and the train–validation gap;
4. choose and restore an early-stopping checkpoint, including the role of patience;
5. reproduce the SGD weight-decay derivation and calculate one complete shrink-plus-data update;
6. explain why AdamW decouples parameter shrinkage from Adam's adaptive gradient path;
7. calculate inverted dropout, explain train/evaluation behavior, and translate the lecture's keep probability into PyTorch's drop probability;
8. decide whether an augmentation is label-preserving for the actual task rather than merely visually plausible;
9. calculate mixup and label-smoothed targets, and distinguish the lecture's smoothing convention from PyTorch's;
10. distinguish accuracy from calibration and run one controlled regularization comparison without touching test.

## The persistent board case: one held five-class classifier

Keep one five-class classifier visible from the opening commitment through the final revisit. Its first report is:

| quantity | value |
|---|---:|
| training accuracy | $99\%$ |
| validation accuracy | $72\%$ on $100$ examples |
| train–validation accuracy gap | $99-72=27$ percentage points |
| test accuracy | **sealed** |

Ask students to commit privately before giving any remedy:

> **What is the first defensible action: deploy from the training score, add many regularizers and select on test, inspect the validation curves while keeping test sealed and then change one lever, or tune directly on test?**

Hold the answer. The lecture must earn it in this order:

1. A $27$-point gap is an observed symptom, not yet a causal verdict; inspect split integrity and the training/validation curves before naming a remedy.
2. Validation may choose a checkpoint or hyperparameter; test cannot.
3. Each remedy changes a different part of the same computation.
4. A remedy stays only if the best validation result improves under a controlled comparison.

The later arithmetic cases are deliberately small views of this same classifier, not claims that every mechanism was trained end to end on one toy model:

- its checkpoint table selects epoch $4$;
- one weight coordinate uses $\theta=5$, data gradient $g=3$, learning rate $\eta=0.1$, and direct decay rate $\lambda_{\mathrm{wd}}=0.2$;
- one hidden vector is $h=[2,4,6,8]$ with keep probability $p_{\text{keep}}=0.5$;
- one label-preserving transform leaves the class unchanged;
- one target has $K=5$ classes and smoothing $\epsilon=0.1$.

At the revisit, require this complete answer: **inspect the curves and split first; if they confirm overfitting, change one matching lever at a time, select and restore using validation, and evaluate test once only after every choice is frozen.**

## Exact 80-minute route

### ⭐ Core · exactly 55 minutes

- L6 bridge, held-classifier commitment, and split roles: 7 min.
- Generalization gap and symptom-based diagnosis: 7 min.
- Learning curves, epoch-$4$ checkpoint, patience, and restore-best semantics: 9 min.
- Weight decay derivation, scalar update, and AdamW distinction: 14 min.
- Inverted dropout expectation, numeric mask, train/evaluation modes, and PyTorch convention: 9 min.
- Controlled experiment, opening revisit, and L8 handoff: 9 min.

### ⭐⭐ Should cover · exactly 15 minutes

- Label-preserving augmentation and one valid/invalid decision: 5 min.
- Five-class mixup target and symbolic cross-entropy: 4 min.
- Label smoothing, the two conventions, and accuracy versus calibration: 6 min.

### ⭐⭐⭐ Optional · exactly 10 minutes

- AdamW parameter groups and restore-best implementation: 5 min.
- Calibration-estimate nuance, notebook traces, and primary-source provenance: 5 min.

The labels sum to **55 core + 15 should-cover + 10 optional = 80 minutes**. The compact handout may contain all three levels; delivery should not give every page equal weight.

### Suggested clock

| Time | Route | Teaching move |
|---|---|---|
| 0–4 | ⭐ | Reuse Lecture 6's last question, show $99\%/72\%$, and collect a private commitment. Do not reveal the remedy. |
| 4–7 | ⭐ | Assign the three split jobs and visibly seal test. |
| 7–14 | ⭐ | Define empirical loss, population risk, the validation estimate, and diagnose from gap plus absolute level. |
| 14–23 | ⭐ | Work the six-epoch table, define patience, and distinguish stop time from winning checkpoint. |
| 23–37 | ⭐ | Derive shrinkage with SGD, calculate $4.60$, and connect coupled $L_2$ to AdamW's decoupled path. |
| 37–46 | ⭐ | Commit to the mask output, derive the expectation, and translate $p_{\text{keep}}$ to PyTorch's $p_{\text{drop}}$. |
| 46–51 | ⭐⭐ | Test one label-preservation claim; make domain knowledge explicit. |
| 51–55 | ⭐⭐ | Calculate the five-class mixup target. |
| 55–61 | ⭐⭐ | Calculate both label-smoothing conventions and separate confidence from calibration. |
| 61–70 | ⭐ | Pre-register one controlled comparison, revisit the opening choice, complete the exit ticket, and hand off to L8. |
| 70–75 | ⭐⭐⭐ | Inspect AdamW parameter groups and restore-best implementation. |
| 75–80 | ⭐⭐⭐ | Select a calibration-estimate caveat or one prediction-first notebook trace, then locate the supporting primary source. |

If the room needs more calculation time, omit the optional block and spend those ten minutes on the checkpoint, weight-decay, and dropout numerics.

## Teach it like this

The lecture is one causal diagnostic, not a catalogue of regularizers:

1. **Establish the visible symptom.** Training accuracy is $99\%$, validation accuracy is $72\%$, and the test result is covered.
2. **Read the symptom before naming a tool.** High train plus much lower validation motivates a split-integrity and curve check; it does not identify which remedy will help or whether $72\%$ meets the task target.
3. **Use time as the first capacity control.** Validation chooses epoch $4$, not the last epoch.
4. **Constrain parameters.** Weight decay adds a direct shrink arrow; AdamW keeps it outside Adam's adaptive gradient path.
5. **Perturb activations.** Dropout samples subnetworks during training while inverted scaling preserves the expected activation.
6. **Constrain behavior under inputs.** Augmentation is valid only when $y(T(x))=y(x)$ for the task; mixup imposes smooth behavior between examples.
7. **Constrain target confidence.** Label smoothing changes the target distribution, not the predicted class rule by itself.
8. **Return to evidence.** Freeze the split, seed, optimizer budget, and metric; change one mechanism; compare best validation; test once.

Use **commit → calculate → reveal → explain** throughout:

- “Which branch does $99/72$ license?” before showing the diagnostic flow.
- “Which epoch wins?” before highlighting $0.78$.
- “Where does the extra $0.10$ come from?” before separating decay and data arrows.
- “What is the masked vector?” before displaying $[4,0,12,0]$.
- “Does this transform preserve the class?” before showing an augmentation.
- “What probability mass goes to each class?” before showing either smoothing vector.

Do not reveal a multiple-choice answer on the same visual state as its question in presentation mode. The handout may collapse builds; the classroom deck should preserve the pause.

## Exact mathematics and board work

### 1 · Train, population, validation, and test

For training data $\{(x_i,y_i)\}_{i=1}^n$,

\[
\mathcal L_{\mathrm{train}}(\theta)
=\frac1n\sum_{i=1}^{n}\ell(f_\theta(x_i),y_i).
\]

The population risk is

\[
\mathcal L_{\mathrm{pop}}(\theta)
=\mathbb E_{(x,y)\sim p_{\mathrm{data}}}
\big[\ell(f_\theta(x),y)\big].
\]

The population quantity is not directly available. During development, a finite validation set estimates unseen performance. A finite test set is another estimate, reserved for the final frozen procedure. Be careful with language: validation does not *equal* population risk, and the test set is not population risk either.

The conceptual generalization gap is

\[
\mathcal L_{\mathrm{pop}}(\theta)-\mathcal L_{\mathrm{train}}(\theta),
\]

while the observable development diagnostic substitutes validation loss for the unavailable population term. For accuracy, say **accuracy gap** and keep its sign convention explicit; for the held classifier it is $99-72=27$ percentage points.

The split rule is non-negotiable:

| split | may change | must not do |
|---|---|---|
| train | parameters $\theta$ | choose the final model by its training score |
| validation | checkpoint, architecture, hyperparameters, thresholds | supply gradient updates |
| test | the final reported estimate | influence any model or reporting choice |

Repeated test inspection transfers information from test into the procedure. Once a result changes a decision, that set has functionally become validation data.

### 2 · Early stopping: choose and restore

Keep the hand calculation identical to the notebook:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| training loss | 1.20 | 0.80 | 0.45 | 0.32 | 0.28 | 0.25 |
| validation loss | 1.25 | 0.98 | 0.82 | **0.78** | 0.84 | 0.97 |

\[
t^\star=\arg\min_t\mathcal L_{\mathrm{val}}(\theta_t)=4.
\]

Epoch $6$ fits train best but does not win model selection. Restore the saved parameters $\theta_4$; do not merely stop at epoch $6$ and keep the last weights.

With the deck's patience $2$, epochs $5$ and $6$ are two consecutive checks without a new validation minimum, so training stops after epoch $6$ and restores epoch $4$. Patience changes when computation ends; it does not redefine the winner. A minimum improvement `min_delta` and a maximum budget must be stated when used. If validation is checked every $k$ updates, the patience clock counts validation checks, not necessarily raw minibatches.

Early stopping is implicit regularization because it restricts which point on the optimization path is reachable without changing the architecture or adding a penalty term.

### 3 · Weight decay: the one reproducible derivation

Use one convention consistently. If

\[
\mathcal J(\theta)
=\mathcal L_{\mathrm{train}}(\theta)+\frac{\lambda}{2}\lVert\theta\rVert_2^2,
\]

then

\[
\nabla_\theta\mathcal J
=\nabla_\theta\mathcal L_{\mathrm{train}}+\lambda\theta,
\]

and an SGD step is

\[
\theta^+
=\theta-\eta(g+\lambda\theta)
=(1-\eta\lambda)\theta-\eta g,
\]

where $g=\nabla_\theta\mathcal L_{\mathrm{train}}$. The factor $1/2$ is the convention used in the deck: it makes the penalty gradient $\lambda\theta$. If an objective is instead written as $\mathcal L+\lambda\lVert\theta\rVert^2$, its penalty gradient is $2\lambda\theta$; never compare numerical coefficients without checking this factor.

For the deck's direct decay coefficient $\lambda_{\mathrm{wd}}$:

\[
\theta=5,\quad g=3,\quad \eta=0.1,\quad \lambda_{\mathrm{wd}}=0.2.
\]

\[
1-\eta\lambda_{\mathrm{wd}}=1-0.1(0.2)=0.98,
\]

\[
\theta^+=0.98(5)-0.1(3)=4.90-0.30=4.60.
\]

Without decay, the update is $5-0.30=4.70$. Thus decay contributes the extra $-0.10$ and the data gradient contributes $-0.30$.

**AdamW nuance:** for plain SGD, an $L_2$ penalty and multiplicative weight decay are equivalent up to coefficient convention. With Adam, putting $\lambda\theta$ inside the gradient feeds it through moment accumulation and coordinate-wise preconditioning. AdamW applies a separate parameter shrink instead:

\[
\theta_{t+1}
=(1-\eta\lambda_{\mathrm{wd}})\theta_t
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

Do not say AdamW is literally optimizing the same regularized objective in the same way as coupled $L_2$, and do not say bias/normalization exclusion is a theorem. Weight matrices are commonly decayed while biases and normalization scale/offset parameters are excluded; the correct parameter grouping is model- and recipe-dependent.

### 4 · Inverted dropout and framework notation

The lecture uses $p_{\mathrm{keep}}$:

\[
m_j\sim\operatorname{Bernoulli}(p_{\mathrm{keep}}),
\qquad
\tilde h_j=\frac{m_jh_j}{p_{\mathrm{keep}}}.
\]

Then

\[
\mathbb E[\tilde h_j]
=p_{\mathrm{keep}}\frac{h_j}{p_{\mathrm{keep}}}
+(1-p_{\mathrm{keep}})0
=h_j.
\]

For

\[
h=[2,4,6,8],\quad
m=[1,0,1,0],\quad
p_{\mathrm{keep}}=0.5,
\]

\[
\tilde h=\frac{m\odot h}{0.5}=[4,0,12,0].
\]

At evaluation, dropout is disabled and the complete $h$ passes through unchanged. The training-time inverse-keep scaling already matched the expected scale; do not multiply again at evaluation.

**PyTorch warning:** `torch.nn.Dropout(p=...)` defines `p` as the probability of **dropping/zeroing**, not keeping. Therefore

\[
p_{\mathrm{PyTorch}}=p_{\mathrm{drop}}=1-p_{\mathrm{keep}}.
\]

With $p_{\mathrm{keep}}=0.5$ the numbers coincide, which can hide the notation error. Use a verbal check such as “keep $0.8$ means `nn.Dropout(p=0.2)`.” Also distinguish elementwise `Dropout` from channelwise spatial dropout variants.

Dropout perturbs activations and gradients; it can slow optimization. It is not guaranteed to help, is not automatically additive with strong augmentation/normalization, and should not be prescribed when the model cannot fit train.

### 5 · Augmentation: preserve the task, not merely the pixels

The intended constraint is

\[
(x,y)\mapsto(T(x),y),
\qquad
f_\theta(T(x))\approx f_\theta(x).
\]

This is justified only when the task label truly survives $T$. For the held five-class object classifier, the deck's defensible policy is: **translate by at most two pixels and reject any transformed sample for which the class-defining object leaves the frame.** A mild crop may preserve “cat,” but a crop that removes the animal does not; a horizontal flip may preserve many natural-image categories but reverse text, handedness, or directional labels.

Use the notebook's oracle diagnostic carefully. On its clean centered-ring rule, a $37^\circ$ rotation gives label agreement $1.000$, while translating by $[0.35,0]$ gives agreement $0.838$. The oracle recomputation diagnoses the transform; it is **not** how augmented training targets are created. In actual augmentation, retain the original label only after domain knowledge supports invariance.

### 6 · Mixup: interpolate both sides

For $\alpha\in[0,1]$,

\[
\tilde x=\alpha x_A+(1-\alpha)x_B,
\qquad
\tilde y=\alpha y_A+(1-\alpha)y_B.
\]

Use $\alpha$ for the mixing coefficient in this section so it is not confused with weight-decay $\lambda$. In the deck's held five-class classifier, take

\[
y_i=[1,0,0,0,0],\qquad y_j=[0,0,0,1,0],\qquad \alpha=0.25.
\]

Then

\[
\tilde y=0.25y_i+0.75y_j=[0.25,0,0,0.75,0],
\]

and for predicted probabilities $p$, $\operatorname{CE}(\tilde y,p)=-0.25\log p_1-0.75\log p_4$. The deck intentionally does not invent a probability vector or decimal loss for this case. Separately, notebook 2 contains a two-class mechanism check with $\alpha=0.7$, target $[0.7,0.3]$, prediction $[0.8,0.2]$, and cross-entropy $0.639$.

Mixup assumes interpolation is meaningful in the chosen input representation and that the soft-label interpolation is appropriate. It is not a universally label-preserving geometric transform, and mixing pixels can create unrealistic examples even when it regularizes successfully.

### 7 · Label smoothing: state the convention

The lecture's **incorrect-classes-only** convention is

\[
y_k^{\mathrm{smooth}}=
\begin{cases}
1-\epsilon, & k=y,\\
\epsilon/(K-1), & k\ne y.
\end{cases}
\]

For $K=5$, $\epsilon=0.1$, and class $1$ correct in the deck,

\[
y^{\mathrm{smooth}}=[0.900,0.025,0.025,0.025,0.025].
\]

The mass check is $0.900+4(0.025)=1$.

PyTorch `CrossEntropyLoss(label_smoothing=epsilon)` instead mixes the one-hot target with a uniform distribution over **all** $K$ classes:

\[
y^{\mathrm{PT}}=(1-\epsilon)y^{\mathrm{onehot}}+\epsilon/K.
\]

At $K=5$, $\epsilon=0.1$, this gives correct-class mass $0.9+0.02=0.92$ and wrong-class mass $0.02$ each:

\[
y^{\mathrm{PT}}=[0.92,0.02,0.02,0.02,0.02].
\]

Neither convention is “the only definition”; the error is switching silently between them. Match the displayed arithmetic to the implementation being used.

Notebook 2 separately uses the same incorrect-classes-only convention but places the $0.9$ mass on class $3$: $[0.025,0.025,0.9,0.025,0.025]$. This is a position change, not a convention change. For its prediction $p_3=0.98$ and $p_k=0.005$ on each wrong class, it obtains hard-label loss $0.020$ and smoothed loss $0.548$; those decimals are notebook evidence, not an extra deck numeric.

Accuracy asks which class wins. Calibration asks whether predictions made with confidence near $q$ are correct about a fraction $q$ of the time. In the deck's one-bin calculation, both versions classify $72$ of $100$ validation examples correctly. Mean confidence $0.90$ gives a bin gap $|0.90-0.72|=0.18$; mean confidence $0.74$ gives $|0.74-0.72|=0.02$. These are single-bin gaps, not full-dataset ECE values. Label smoothing often reduces extreme confidence, but improved confidence histograms do not guarantee better calibration under distribution shift; measure the calibration metric relevant to the task. Smoothing can also be undesirable when preserving probability information for distillation or uncertainty-sensitive decisions matters.

## Controlled experiment protocol

For every remedy, write the comparison before running it:

1. **Baseline:** fixed data split, preprocessing, seed policy, architecture, optimizer, update budget, and evaluation code.
2. **One change:** one decay coefficient, dropout rate, augmentation policy, mixup strength, smoothing value, or stopping rule.
3. **Recorded evidence:** training curve, validation curve, best checkpoint, relevant mechanism measure (for example weight norm), and wall/update budget.
4. **Decision:** keep or reject using validation; repeat promising settings across several seeds when variance matters.
5. **Final report:** freeze every decision, restore the selected checkpoint, and evaluate test once.

Do not compare a regularized run trained for twice as many updates with an unregularized baseline and attribute the difference only to regularization. Do not move five knobs together. Do not select by the lowest norm, prettiest boundary, or smallest training loss.

## Recurring diagnostic: symptom → suspect → test

| observed symptom | first suspect | smallest informative test | first move if confirmed |
|---|---|---|---|
| train high, validation much lower | overfitting / split mismatch | inspect both curves and split construction | test one regularizer on validation |
| train and validation both poor | underfitting, optimization, data pipeline | overfit one small batch; inspect inputs/labels | repair fit; reduce regularization |
| validation improves then rises | late overfitting | restore saved minimum; verify epoch clock | early stopping |
| boundary changes under a valid nuisance | missing invariance / excessive sensitivity | transform held-out inputs and compare predictions | valid augmentation, then validate |
| weight norm grows while validation worsens | brittle high-norm solution may be involved | decay sweep with identical budget | modest weight decay |
| train/eval predictions differ unexpectedly | dropout or normalization mode bug | same batch under explicit `.train()`/`.eval()` | repair mode handling |
| confidence is extreme at unchanged accuracy | target/logit confidence issue | reliability diagram and task-relevant calibration score | test smoothing or post-hoc calibration |
| repeated test improvements guide choices | evaluation leakage | audit every choice that saw test | obtain/redefine a truly sealed test |

The table is a hypothesis generator, not a lookup oracle. “Suspect” must be followed by a controlled test.

## Notebook choreography: exactly two follow-alongs

Use the public Colab links; do not add a third notebook.

### Notebook 1 · Diagnose overfitting and stop at the right checkpoint

[`01_overfitting_and_early_stopping.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L07/01_overfitting_and_early_stopping.ipynb)

**Before running:**

1. Assign train, validation, and test roles for the $24/24/24$ seeded split.
2. Predict which of degrees $1$, $5$, and $10$ will underfit or overfit.
3. Calculate the six-epoch checkpoint and state why epoch $6$ cannot win.
4. Predict the directions of training and validation curves after the minimum for the degree-$24$ gradient-descent fit.

**Reveal only after commitments:**

- degree $1$: train MSE $0.3117$, validation MSE $0.2425$;
- degree $5$: train MSE $0.0308$, validation MSE $0.0428$;
- degree $10$: train MSE $0.0237$, validation MSE $4.3282$;
- hand table: best checkpoint epoch $4$, validation loss $0.78$;
- degree-$24$ run: best epoch $1640$, best validation MSE $0.1167$, last validation MSE $0.1756$;
- after every choice is frozen: best-checkpoint test MSE $0.1111$, printed once.

**Explain:** degree alone is not the diagnosis; cite train and validation values plus the curve shape. The feature standardization is fit on train and reused on validation/test. Epoch $1$ means the first parameter update, so the clock is one-based. The notebook never compares candidate checkpoints using test.

### Notebook 2 · What each remedy changes

[`02_dropout_from_scratch.ipynb`](https://colab.research.google.com/github/nipunbatra/dl-teaching/blob/master/notebooks/L07/02_dropout_from_scratch.ipynb)

**Before running:**

1. Predict which decay setting has the smallest weight norm and whether the strongest decay must have the best validation loss.
2. Decide whether rotation and translation preserve a centered-ring label.
3. Calculate the scalar weight update and fixed dropout mask by hand.
4. Calculate the mixup and smoothed targets before Python prints them.

**Reveal only after commitments:**

| decay | train CE | validation CE | $\lVert w\rVert$ |
|---:|---:|---:|---:|
| $0.00$ | $0.071$ | $0.596$ | $4.921$ |
| $0.03$ | $0.165$ | **$0.456$** | $2.195$ |
| $0.20$ | $0.305$ | $0.482$ | $0.928$ |

- rotation label agreement $=1.000$; translation agreement $=0.838$;
- shrink factor $0.98$, post-shrink weight $4.90$, data contribution $-0.30$, new weight $4.60$;
- fixed dropout mask $[4,0,12,0]$; Monte Carlo mean approximately $[2.011,4.009,6.045,8.046]$; evaluation output $[2,4,6,8]$;
- mixup target $[0.7,0.3]$, soft cross-entropy $0.639$;
- incorrect-classes-only smoothing target $[0.025,0.025,0.9,0.025,0.025]$, hard loss $0.020$, smoothed loss $0.548$.

**Explain:** decay $0.20$ achieves the smallest norm but not the best validation CE; validation selects $0.03$ among these runs. Weight decay and augmentation use the persistent ring case. Dropout, mixup, and smoothing are isolated arithmetic/mechanism checks, not evidence that those methods improve the ring classifier. The notebook uses only NumPy and Matplotlib, fixed RNG seeds, retained outputs, and no network downloads.

## Heads-up for the instructor

- **A validation set is used repeatedly by design.** It can itself become overfit after extensive search; nested validation, a fresh holdout, or conservative search may be needed for a high-stakes estimate. This does not license tuning on test.
- **A gap alone is incomplete.** $65\%/63\%$ is a small gap and a bad model; $99\%/72\%$ is a large gap after successful fitting.
- **Patience is not the checkpoint.** Stop after the configured non-improvement count, then restore the minimum-validation parameters.
- **Penalty coefficients have conventions.** $\lambda\lVert\theta\rVert^2$ differentiates to $2\lambda\theta$; $(\lambda/2)\lVert\theta\rVert^2$ differentiates to $\lambda\theta$. Framework `weight_decay` commonly names the direct decay coefficient.
- **AdamW is not “Adam plus the same $L_2$ term.”** Its point is decoupling from adaptive moment/preconditioner behavior.
- **PyTorch dropout `p` drops.** The lecture's $p_{\mathrm{keep}}$ keeps. Translate every time.
- **Expected activation is not identical activation.** Inverted dropout preserves the mean under the mask distribution, not each realization or the variance.
- **Augmentation is a task claim.** A transform valid for object identity may be invalid for pose, localization, OCR, medical laterality, or color-defined labels.
- **Mixup is not ordinary augmentation.** It interpolates both input and target and assumes the interpolation is meaningful enough to be useful.
- **Label smoothing has multiple conventions.** The lecture vector $0.9/0.025$ differs from PyTorch's $0.92/0.02$ at the same stated $K$ and $\epsilon$.
- **Lower confidence is not automatically calibrated probability.** Measure calibration, and distinguish in-distribution calibration from uncertainty under shift.

## Where students stumble (and the fix)

- **“Training accuracy is high, so the model is good.”** Cover validation and test separately; ask what information each number contains.
- **“The smallest train–validation gap wins.”** A small gap can still accompany poor absolute performance; require both the level and the gap before prescribing a remedy.
- **“Early stopping keeps the last weights.”** Physically label a saved box $\theta_4$ and restore it after the patience clock fires.
- **“More decay is safer.”** Use the notebook table: norm keeps falling from $2.195$ to $0.928$, while validation worsens from $0.456$ to $0.482$.
- **“Dropout $p=0.8$ keeps $80\%$ in PyTorch.”** Translate aloud: lecture keep $0.8$ means PyTorch drop $0.2$.
- **“At evaluation, multiply by keep probability.”** Inverted dropout already scales during training; evaluation is identity.
- **“Every crop/flip is free data.”** Ask whether an oracle for the actual label would agree before and after the transform.
- **“Mix only the images and keep one hard label.”** Write $\tilde x$ and $\tilde y$ on the same line.
- **“Label smoothing always means correct class $1-\epsilon$.”** Show the lecture and PyTorch vectors side by side.
- **“Calibration equals accuracy.”** Ask two questions: which class wins, and among $80\%$ predictions, how often is the model right?

## If a student asks…

- **“Why not use test whenever we need an honest number?”** Because the first inspection can be honest, but changing anything in response makes the complete procedure depend on test. Validation exists to absorb those decisions; test evaluates the frozen procedure.
- **“Can validation also be overfit?”** Yes. Large hyperparameter searches and repeated human iteration can overfit validation. Use restrained search, multiple seeds, nested validation, or a fresh holdout when the stakes justify it.
- **“Is early stopping just saving compute?”** It can save compute, but its regularizing effect is model selection along the optimization path. Patience and a budget govern stopping; the minimum validation checkpoint governs selection.
- **“Is weight decay the same as a Gaussian prior?”** Coupled $L_2$ in a precisely defined objective has the MAP interpretation from Lecture 1. Decoupled AdamW is an optimizer update rule whose behavior is not identical to passing that penalty through Adam.
- **“Should I always use AdamW?”** It is a strong default when using Adam-family optimizers and decay, but decay strength and parameter exclusions are empirical recipe choices. A no-decay baseline is still necessary.
- **“Does dropout average many networks?”** It is useful intuition: training samples subnetworks with shared weights. Do not claim exact Bayesian model averaging; evaluation uses a deterministic scaled-network approximation.
- **“Why can dropout hurt a CNN?”** It adds noise and may be redundant with convolutional structure, normalization, and strong augmentation. Measure rather than importing an MLP default.
- **“How do I know an augmentation is valid?”** State the nuisance the task should ignore, inspect transformed examples, and test label agreement or domain constraints. If the transformation can alter the target, do not retain the old label blindly.
- **“Is mixup label preserving?”** Not in the ordinary hard-label sense. It defines a soft target between two examples and asks the model to vary smoothly along that interpolation.
- **“Does label smoothing improve calibration?”** Often it reduces overconfidence, but not universally and not for every calibration metric or shift. Evaluate accuracy and calibration separately.

## If you are short on time

Teach the **55-minute core** and omit the entire optional block. If less than 55 minutes remain, retain these in order:

1. $99\%/72\%$ commitment and train/validation/test roles;
2. epoch-$4$ early-stopping calculation and restore-best rule;
3. the SGD shrinkage derivation and $4.60$ numeric, followed by one sentence on AdamW decoupling;
4. inverted-dropout expectation, $[4,0,12,0]$, and PyTorch's drop-probability convention;
5. one valid/invalid augmentation decision;
6. controlled validation comparison, opening revisit, and L8 handoff.

Assign mixup and label-smoothing arithmetic as follow-along work. Cut parameter-group code, calibration-estimate details, notebook trace reveals, provenance discussion, and extended Q&A first. Never cut the sealed-test protocol, best-checkpoint restoration, or framework convention warnings.

## Primary sources and provenance

- Prechelt (1998), *Early Stopping—but When?*
- Srivastava et al. (2014), *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*.
- Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization*.
- Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization*.
- Szegedy et al. (2016), *Rethinking the Inception Architecture for Computer Vision* (label smoothing).
- Guo et al. (2017), *On Calibration of Modern Neural Networks*.
- PyTorch documentation for `torch.optim.AdamW`, `torch.nn.Dropout`, and `torch.nn.CrossEntropyLoss(label_smoothing=...)`; framework conventions should be checked against the version used in class.

The synthetic learning curves and notebook results are mechanism demonstrations, not benchmark claims. Quote the exact displayed data when drawing a conclusion, and avoid treating a decorative smooth boundary or norm curve as empirical evidence unless it was generated by the shown computation.

## Closing line

> **Optimization fits the training set. Regularization chooses a stable fitting solution. Validation is the judge—and test speaks once. Next, convolution will encode one useful image assumption directly in the architecture.**
