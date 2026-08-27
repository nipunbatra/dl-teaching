# Lecture 7 · Generalization and Regularization · Teaching Guide

Instructor companion for `lecture7/L7-regularization.typ`. The audience has completed Lectures 1–6 and an introductory ML course.

## The lecture in two sentences

Lectures 3–6 made optimization reliable; Lecture 7 asks what happens when training loss keeps improving but performance on new data does not. Regularization states a preference at the objective, data, model, target, or training-loop level; validation chooses its strength, and a sealed test set reports the final result once.

Repeat this line:

> **Regularization decides what the model does where the data is silent. You state the preference; validation sets the knob.**

## The two teaching cases

### 1. Exact 10-point regression spine

The diagnosis, weight-decay, early-stopping, and input-jitter figures are computed in Typst from one seeded dataset: `y = sin(πx) + noise`, ten equispaced training inputs, a degree-9 polynomial feature map, nine validation midpoints, and a separate dense test draw. This case makes the mathematics inspectable.

Values worth saying aloud:

| quantity | value |
|---|---:|
| interpolant train MSE | 0.0000 |
| interpolant validation MSE | 1.70 |
| observation-noise variance | 0.0625 |
| interpolant weight norm | about 465 |
| validation-selected weight decay | λ = 10⁻³; validation MSE 0.065 |
| validation-selected stopping time | 5,000 updates; validation MSE 0.061 |
| eight-copy input jitter | validation MSE 0.048 |

The loss convention is fixed throughout:

\[
\mathcal L_{\text{data}}(w)=\frac{1}{2n}\lVert Xw-y\rVert_2^2,
\qquad
\mathcal J(w)=\mathcal L_{\text{data}}(w)+\frac{\lambda_{\rm wd}}2\lVert w\rVert_2^2.
\]

The companion `lecture7/diagrams/verify_numbers.typ` independently checks the displayed regression numbers.

Treat the same-label input jitter as a deliberately imposed **local smoothness prior**, not an exact invariance of `sin(πx)`. It biases the signal slightly; validation decides whether the variance reduction is worth that bias.

### 2. Measured CIFAR-10 practical case

The later slides use real CIFAR-10 images and a controlled PyTorch experiment: 2,000 stratified training examples, 5,000 validation examples, the untouched 10,000-example test split, one 666,538-parameter CNN, one initialization, one minibatch order, and 35 epochs. Each run changes one regularizer only; validation loss selects its checkpoint.

| condition | selected epoch | validation accuracy | test accuracy |
|---|---:|---:|---:|
| baseline | 8 | 42.74% | 42.99% |
| AdamW decay | 8 | 43.00% | 43.05% |
| crop + horizontal flip | 28 | 53.60% | 54.62% |
| dropout | 15 | 46.58% | 46.79% |
| label smoothing | 11 | 45.44% | 44.82% |

The baseline ends at 99.55% training accuracy and 44.24% validation accuracy: a clean overfitting demonstration. Say the caveats with the result: the tested AdamW setting (`weight_decay = 0.01`) is neutral in this particular recipe, and one fixed seed is a teaching case, not a benchmark.

The five-condition suite and selection rules were predeclared. Within each fixed condition, the checkpoint was frozen using validation loss before its one reporting-only test evaluation; test results were never inputs to method or hyperparameter selection.

Frame this as a predeclared *methods study*: several frozen candidates can be reported in one final test-analysis stage, provided nothing is changed afterward. In an ordinary deployment workflow, the test report would usually contain only the single validation-selected recipe.

Reproduction artifacts live in `lecture7/diagrams/cifar_regularization_experiment.py` and `lecture7/evidence/`.

## Learning contract

Students should leave able to:

1. read training and validation curves using level, gap, and trend;
2. state the distinct roles of train, validation, and sealed test splits;
3. explain why zero training loss cannot choose among interpolating solutions;
4. derive `w ← (1 − ηλ)w − η∇L_data(w)` for SGD;
5. explain why an L₂ penalty and weight decay coincide for plain SGD but not adaptive preconditioning, motivating AdamW;
6. save and restore the validation-best checkpoint with patience and `min_delta`;
7. write a train-only augmentation pipeline, visually audit it, and transform spatial supervision with the image;
8. compute MixUp/CutMix soft targets and use them with cross-entropy;
9. distinguish dropout's `p_drop` from `q_keep`, derive inverted scaling, and explain `train()` versus `eval()`;
10. perform MC Dropout without changing BatchNorm state, averaging probabilities rather than logits;
11. state PyTorch's all-K label-smoothing convention and explain why soft-target methods can interact;
12. run a controlled ablation and use the test set only after every choice is frozen.

## Recommended 90-minute route

| minutes | arc | landing points |
|---:|---|---|
| 0–12 | Diagnose | disagreeing curves → three split contract → 0.0000 train / 1.70 validation → bias–variance → two interpolants |
| 12–27 | Weight decay | exact convention → coefficient explosion → λ sweep → SGD shrinkage → Adam distortion → AdamW groups → brief L₁ |
| 27–38 | Early stopping | train/validation curves → `t*` → save/restore code → spectral intuition → capacity checkpoint |
| 38–55 | Augmentation | real CIFAR views → paired-transform objective → valid/invalid invariances → torchvision v2 → batch audit → MixUp/CutMix |
| 55–72 | Dropout | co-adaptation → Bernoulli mask → expectation/variance → code and placement → state-mode table → MC Dropout bridge and safe code |
| 72–80 | Label smoothing | finite confidence optimum → all-K convention → PyTorch code → interactions with MixUp/CutMix |
| 80–90 | Synthesis | taxonomy → staged recipe → symptom table → measured CIFAR ablation → sealed synthetic test → exit ticket |

For an 80-minute class, cut the optional spectral-filter equivalence, L₁, specialized-regularizer table, and one of the two CIFAR evidence slides. Do not cut the split contract, AdamW distinction, restore-best code, augmentation validity check, dropout state-mode table, or MC Dropout bridge.

## Narrative and delivery

- Start with the disagreement between the curves, not a catalogue of techniques.
- Keep the regression spine active through early stopping: same data, same flexible model, different intervention.
- Switch explicitly to the CIFAR case for image-specific practice. The image gallery should prompt the question “would a domain expert keep this label?”
- At every method, name three things: **site**, **knob**, and **measurement**.
- Treat question slides as commitment devices. Students answer before the adjacent answer slide.
- The final CIFAR ablation is deliberately honest: augmentation wins clearly, dropout and smoothing help modestly, and the tested decay value does not help under this fixed recipe.

## Board work

### Weight decay

\[
\nabla \mathcal J(w_t)=g_t+\lambda w_t
\quad\Longrightarrow\quad
w_{t+1}=(1-\eta\lambda)w_t-\eta g_t.
\]

With `w = 5`, `g = 3`, `η = 0.1`, `λ = 0.2`: shrink to 4.90, take the −0.30 data step, finish at 4.60.

For adaptive methods, putting `λw` through the preconditioner changes the intended proportional shrink coordinate by coordinate. AdamW applies decay outside the adaptive data-gradient update.

### Early stopping

\[
t^\star=\operatorname*{arg\,min}_t \mathcal L_{\rm val}(w_t).
\]

Emphasize “restore the best state,” not merely “break the loop.” To resume training rather than evaluate, also save optimizer, scheduler, scaler, epoch, and RNG state.

### Augmentation

\[
(x',y')=T(x,y),\qquad
\min_\theta\;\mathbb E_{(x,y)}\mathbb E_{T\sim\mathcal T}
\big[\ell(f_\theta(x'),y')\big].
\]

Classification often keeps the class index fixed. Detection, segmentation, and keypoint tasks must move their targets with the pixels.

### MixUp

For a 70/30 cat–dog blend:

\[
\tilde y=0.70y_{\rm cat}+0.30y_{\rm dog},\qquad
\ell=-0.70\log p_{\rm cat}-0.30\log p_{\rm dog}.
\]

Torchvision's v2 MixUp and CutMix operate after batching and return probability targets accepted directly by `torch.nn.functional.cross_entropy`.

### Inverted dropout

Let `p_drop = p` and `q_keep = 1 − p`:

\[
m_i\sim\operatorname{Bernoulli}(q),\qquad
\tilde h_i=\frac{m_i}{q}h_i,qquad
\mathbb E[\tilde h_i]=h_i,qquad
\operatorname{Var}(\tilde h_i)=\frac{1-q}{q}h_i^2.
\]

PyTorch's `nn.Dropout(p=...)` uses the drop probability. In ordinary evaluation it is the identity because retained activations were already scaled during training.

### Label smoothing

PyTorch uses the all-K mixture:

\[
\tilde y=(1-\varepsilon)y+\frac{\varepsilon}{K}\mathbf 1.
\]

For `K = 5`, `ε = 0.1`, the correct class receives 0.92 and each other class 0.02. This differs from redistributing ε only among the wrong classes.

## The state-mode slide is non-negotiable

| control | dropout | BatchNorm | autograd |
|---|---|---|---|
| `model.train()` | random masks | batch statistics; buffers update | unchanged |
| `model.eval()` | identity | stored statistics | unchanged |
| `torch.inference_mode()` | unchanged | unchanged | disabled |

`eval()` and `inference_mode()` solve different problems. This distinction is the prerequisite for the next lecture.

For MC Dropout:

1. call `model.eval()` so BatchNorm remains frozen;
2. switch only `nn.Dropout*` modules to training mode;
3. run multiple forwards inside `torch.inference_mode()`;
4. average probabilities, not logits;
5. restore `model.eval()` afterward.

Do not reuse code that calls `model.train()` globally at inference. Functional `F.dropout` also needs an explicit MC-inference flag because it is not discoverable as a module.

## Practical code checkpoints

- AdamW parameter groups visibly separate decayed weights from biases and one-dimensional scale/shift parameters. Present that exclusion as a common recipe, not a theorem.
- Early stopping resets patience on improvement, increments it otherwise, deep-copies the winning state, and restores it.
- Stochastic train transforms and deterministic evaluation transforms are separate objects.
- Inspect a grid of transformed images before training.
- MixUp/CutMix happen after batching and produce soft targets.
- Dropout is placed before a learned downstream layer, not on final logits.
- Evaluation reports accuracy plus NLL/calibration when confidence quality matters.
- If MixUp/CutMix already softens targets, test label smoothing as a separate ablation; stacking can over-soften.

## Common misconceptions

- **“A small train–validation gap means success.”** Both losses may simply be high. Read their absolute levels and trends too.
- **“Training loss chooses the regularizer.”** Training fits parameters; validation selects the recipe.
- **“More regularization is safer.”** Every knob can cross from variance reduction into damaging bias.
- **“Adam's L₂ penalty is AdamW.”** Coupled L₂ is preconditioned; AdamW's decay is decoupled.
- **“Early stopping means stopping at the final epoch.”** The selected checkpoint must be restored.
- **“Flips are universally safe.”** A horizontal flip changes `b` to `d` and can violate anatomy or text semantics.
- **“`eval()` disables gradients.”** It changes module behavior; gradient tracking is separate.
- **“MC Dropout means `model.train()` at test time.”** That also changes BatchNorm and is usually wrong.
- **“Label smoothing always improves calibration.”** Its effect depends on the data and recipe; measure it.
- **“The best validation method must win on this one test draw.”** Validation is noisy; the sealed test estimates the frozen selection procedure.

## Notebook and interactive choreography

- After early stopping: `notebooks/L07/01_overfitting_and_early_stopping.ipynb`. Ask students to predict the validation minimum and a suitable patience before running it.
- The dropout notebook contains useful from-scratch mechanics, but the slides are canonical for `p_drop/q_keep`, MC inference, and PyTorch's all-K smoothing convention.
- In-deck interactives cover the regularization map, weight decay, augmentation/MixUp, dropout, and calibration.

## Handoff to the next lecture

Students now know why different dropout masks produce different subnetworks and how to control module state safely. The next lecture keeps dropout stochastic at inference, summarizes repeated probability vectors with a predictive mean and uncertainty measures, and then studies calibration and failure modes.

Closing line:

> “The mask began as a training regularizer. Keep it stochastic at inference, and one network becomes a distribution of predictions.”

## Primary sources

Zhang et al. (2017), rethinking generalization · Loshchilov and Hutter (2019), AdamW · Srivastava et al. (2014), dropout · Gal and Ghahramani (2016), MC Dropout · Zhang et al. (2018), MixUp · Yun et al. (2019), CutMix · Cubuk et al. (2020), RandAugment · Szegedy et al. (2016), label smoothing · Müller et al. (2019), label-smoothing tradeoffs · Bishop (1995), training with noise.
