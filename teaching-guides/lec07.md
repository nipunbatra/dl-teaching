# Lecture 7 · Generalization and Regularization · Teaching Guide

Instructor companion for `lecture7/L7-regularization.typ`. The audience has completed Lectures 1–6 and an introductory ML course.

## The lecture in two sentences

Lecture 7 asks what happens when training loss keeps improving but performance on new data does not. Regularization states a preference at the objective, data, model, target, or training-loop level; validation chooses its strength, and a sealed test set reports the final result once.

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

### 2. High-resolution visual examples, then measured CIFAR-10 evidence

The augmentation, task-conditioned target, MixUp/CutMix, and dropout-analogy slides use course-owned high-resolution teaching photographs so students can inspect the operation on a projector. The cyclist, runner, and duck examples receive exact boxes, keypoints, flips, and masks in Typst; the cat/dog blends use exact pixel arithmetic. These are illustrations, not experiment inputs. The later controlled PyTorch experiment returns explicitly to CIFAR-10: 2,000 stratified training examples, 5,000 validation examples, the untouched 10,000-example test split, one 666,538-parameter CNN, one initialization, one minibatch order, and 35 epochs. Each run changes one regularizer only; validation loss selects its checkpoint.

| condition | selected epoch | validation accuracy | test accuracy |
|---|---:|---:|---:|
| baseline | 8 | 42.74% | 42.99% |
| weight decay (implemented with AdamW) | 8 | 43.00% | 43.05% |
| crop + horizontal flip | 28 | 53.60% | 54.62% |
| dropout | 15 | 46.58% | 46.79% |
| label smoothing | 11 | 45.44% | 44.82% |

The baseline ends at 99.55% training accuracy and 44.24% validation accuracy: a clean overfitting demonstration. Say the caveats with the result: the tested weight-decay setting (`weight_decay = 0.01`) is neutral in this particular recipe, and one fixed seed is a teaching case, not a benchmark. The experiment uses AdamW, but the Adam-versus-AdamW distinction belongs to the optional appendix rather than the main route.

The five-condition suite and selection rules were predeclared. Within each fixed condition, the checkpoint was frozen using validation loss before its one reporting-only test evaluation; test results were never inputs to method or hyperparameter selection.

Frame this as a predeclared *methods study*: several frozen candidates can be reported in one final test-analysis stage, provided nothing is changed afterward. In an ordinary deployment workflow, the test report would usually contain only the single validation-selected recipe.

Reproduction artifacts live in `lecture7/diagrams/cifar_regularization_experiment.py` and `lecture7/evidence/`.

## Learning contract

Students should leave able to:

1. read training and validation curves using level, gap, and trend;
2. state the distinct roles of train, validation, and sealed test splits;
3. explain why zero training loss cannot choose among interpolating solutions;
4. derive `w ← (1 − ηλ)w − η∇L_data(w)` for SGD;
5. implement the same L₂-regularized SGD update without applying the penalty twice;
6. save and restore the validation-best checkpoint with patience and `min_delta`;
7. write an explicit target contract (A_\tau(x,y)=(g_\tau(x),h_\tau(y))), including boxes, keypoints, categorical masks, timestamps, and signed regression targets;
8. write and audit a train-only pipeline, including the two RandAugment knobs and its image/video-only TorchVision limitation;
9. compute MixUp/CutMix soft targets, identify domains where raw-input interpolation needs adaptation, and use probability targets with cross-entropy;
10. distinguish dropout's `p_drop` from `q_keep`, decode every term in inverted dropout, and explain `train()` versus `eval()`;
11. state PyTorch's all-K label-smoothing convention and explain how it limits pressure toward extreme logits;
12. run a controlled ablation and use the test set only after every choice is frozen.

## Recommended 90-minute route

| minutes | arc | landing points |
|---:|---|---|
| 0–11 | Diagnose | disagreeing curves → three split contract → 0.0000 train / 1.70 validation → bias–variance → two interpolants |
| 11–25 | Weight decay | exact convention → coefficient explosion → λ sweep → SGD shrinkage geometry → scalar derivation → PyTorch SGD → brief L₁ |
| 25–35 | Early stopping | train/validation curves → `t*` → save/restore code → why validation rises → training time as capacity |
| 35–58 | Augmentation | exact-vs-invalid invariances → target map (h_\tau) → boxes/keypoints/masks → input jitter → CIFAR v2 recipe → repeated-view audit → RandAugment (N,M) → MixUp/CutMix and domain test |
| 58–75 | Dropout | road-closure story → one fixed network → worked hidden vector → fresh masks → divide by `q_keep` → mean-preserving calculation → evaluation identity → state modes and placement |
| 75–82 | Label smoothing | one-hot widens logit gaps → “confident enough” target → finite preferred gap → PyTorch code and limits |
| 82–90 | Synthesis | taxonomy → staged recipe → symptom table → measured CIFAR ablation → sealed synthetic test → exit ticket |

For an 80-minute class, cut L₁, the specialized-regularizer table, and one of the two CIFAR evidence slides. Do not cut the split contract, restore-best code, augmentation validity check, dropout worked example, inverted-scaling explanation, or state-mode table. AdamW remains an optional appendix; predictive uncertainty and MC Dropout belong to the later uncertainty lecture.

## Narrative and delivery

- Start with the disagreement between the curves, not a catalogue of techniques.
- Keep the regression spine active through early stopping: same data, same flexible model, different intervention.
- Switch explicitly to the high-resolution visual examples for augmentation practice. The gallery should prompt “would a domain expert keep this label?” Use the cyclist to make the central distinction: the same flip preserves an object class, changes travel direction, and moves a detection box. Before the CIFAR transform code, state that the deck is returning to the 32×32 recipe used in the measured ablation.
- Treat RandAugment and Mixup as different ideas. RandAugment composes ordinary per-image operations with two policy knobs; Mixup imposes linear behaviour between examples and changes the target distribution.
- Name the dropout group-project image as an analogy, then return immediately to exact masks and equations. Say explicitly that surviving features still collaborate; dropout reduces reliance on a fixed coalition rather than creating one-unit classifiers.
- Keep dropout here as a training-time regularizer with deterministic evaluation. Do not detour into MC Dropout or uncertainty estimation.
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

With `w = 5`, `g = 3`, `η = 0.1`, `λ = 0.2`: shrink to 4.90, take the −0.30 data step, finish at 4.60. In code, use either optimizer-side `weight_decay` or add the penalty to the loss—not both.

### Optional appendix: AdamW

Only take this detour if students need the adaptive-optimizer distinction. Coupled L₂ sends `g + λw` through Adam's preconditioner; AdamW keeps the proportional shrink separate from Adam's data-gradient update. This distinction is not a main-route learning requirement.

### Early stopping

\[
t^\star=\operatorname*{arg\,min}_t \mathcal L_{\rm val}(w_t).
\]

Emphasize “restore the best state,” not merely “break the loop.” To resume training rather than evaluate, also save optimizer, scheduler, scaler, epoch, and RNG state.

### Augmentation

\[
A_\tau(x,y)=\bigl(g_\tau(x),h_\tau(y)\bigr),\qquad
\min_\theta\;\mathbb E_{(x,y)}\mathbb E_{T\sim\mathcal T}
\big[\ell(f_\theta(g_\tau(x)),h_\tau(y))\big].
\]

Classify every transform before coding it: keep the target, move/change it by a known map, or reject/relabel the sample. Detection boxes use the same sampled geometry as the image; keypoint flips may also permute semantic left/right indices; categorical masks use nearest-neighbour resampling so class IDs remain class IDs.

For input jitter, write

\[
\varepsilon\sim\mathcal N(0,\sigma^2I),\qquad
(x_i,y_i)\mapsto(x_i+\varepsilon,y_i).
\]

This is a local-constancy approximation controlled by \(\sigma\), not a claim of exact invariance.

### RandAugment

`num_ops = N` is the number of randomly selected operations applied in sequence. `magnitude = M` is an index into operation-specific scales, not degrees or percent. In TorchVision, keep the image in `uint8` through `v2.RandAugment`, then convert/scale and normalize; the transform supports images/videos, not boxes, masks, or keypoints.

### MixUp

For a 70/30 cat–dog blend:

\[
\tilde y=0.70y_{\rm cat}+0.30y_{\rm dog},\qquad
\ell=-0.70\log p_{\rm cat}-0.30\log p_{\rm dog}.
\]

Torchvision's v2 MixUp and CutMix operate after batching and return probability targets accepted directly by `torch.nn.functional.cross_entropy`.

State the domain test before code: the blended representation must be defined, the soft target must encode the intended supervision, and validation gains must survive a domain audit. Raw-input Mixup is not plug-and-play for token IDs, graphs/molecules, spatial targets, or control targets whose actions do not interpolate with blended scenes.

### Inverted dropout

Let `p_drop = p` and `q_keep = 1 − p`:

\[
m_i\sim\operatorname{Bernoulli}(q),\qquad
\tilde h_i=\frac{m_i}{q}h_i,\qquad
\mathbb E[\tilde h_i]=h_i.
\]

PyTorch's `nn.Dropout(p=...)` uses the drop probability. In ordinary evaluation it is the identity because retained activations were already scaled during training.

### Label smoothing

PyTorch uses the all-K mixture:

\[
\tilde y=(1-\varepsilon)y+\frac{\varepsilon}{K}\mathbf 1.
\]

For `K = 5`, `ε = 0.1`, the correct class receives 0.92 and each other class 0.02. One-hot cross-entropy keeps rewarding a wider correct-versus-wrong logit gap; the smoothed target stops pushing when the probabilities match these finite targets. If the four wrong-class logits are equal, the preferred gap is `z_b − z_j = ln(0.92/0.02) ≈ 3.83`. This differs from redistributing ε only among the wrong classes.

## The state-mode slide is non-negotiable

| control | dropout | BatchNorm | autograd |
|---|---|---|---|
| `model.train()` | random masks | batch statistics; buffers update | unchanged |
| `model.eval()` | identity | stored statistics | unchanged |
| `torch.inference_mode()` | unchanged | unchanged | disabled |

`eval()` and `inference_mode()` solve different problems. For ordinary validation and testing, use both: `model.eval()` makes dropout deterministic and freezes BatchNorm behavior, while `torch.inference_mode()` disables gradient tracking.

## Practical code checkpoints

- Early stopping resets patience on improvement, increments it otherwise, deep-copies the winning state, and restores it.
- Stochastic train transforms and deterministic evaluation transforms are separate objects.
- Inspect a grid of transformed images before training.
- RandAugment comes before float conversion/normalization; visually audit draws and validate `num_ops` and `magnitude`.
- MixUp/CutMix happen after batching and produce soft targets.
- The Mixup/CutMix loader is shuffled; class-ID labels `(B,)` become probability targets `(B, K)`.
- Dropout is placed before a learned downstream layer, not on final logits.
- Evaluation reports accuracy plus NLL/calibration when confidence quality matters.
- If MixUp/CutMix already softens targets, test label smoothing as a separate ablation; stacking can over-soften.

Optional appendix checkpoint: if AdamW is shown, present parameter-group exclusions as a common recipe rather than a theorem.

## Common misconceptions

- **“A small train–validation gap means success.”** Both losses may simply be high. Read their absolute levels and trends too.
- **“Training loss chooses the regularizer.”** Training fits parameters; validation selects the recipe.
- **“More regularization is safer.”** Every knob can cross from variance reduction into damaging bias.
- **“Early stopping means stopping at the final epoch.”** The selected checkpoint must be restored.
- **“Flips are universally safe.”** A horizontal flip changes `b` to `d` and can violate anatomy or text semantics.
- **“An augmentation library knows how my labels should change.”** The task defines (h_\tau); write and audit it explicitly.
- **“RandAugment discovers the right invariances.”** It reduces policy search to a small set of knobs; the operation space is still a domain claim.
- **“Mixup is another invariance transform.”** It changes both input and target and imposes a linearity prior between examples.
- **“Dropout deletes weights.”** It masks activations for one training forward pass; the stored network and its weights remain intact.
- **“`eval()` disables gradients.”** It changes module behavior; gradient tracking is separate.
- **“Label smoothing always improves calibration.”** Its effect depends on the data and recipe; measure it.
- **“The best validation method must win on this one test draw.”** Validation is noisy; the sealed test estimates the frozen selection procedure.

Optional appendix pitfall: coupled L₂ in Adam is not AdamW; AdamW keeps decay separate from the adaptive data-gradient update.

## Notebook and interactive choreography

- After early stopping: `notebooks/L07/01_overfitting_and_early_stopping.ipynb`. Ask students to predict the validation minimum and a suitable patience before running it.
- The dropout notebook contains useful from-scratch mechanics, but the slides are canonical for `p_drop/q_keep`, inverted scaling, and ordinary train/evaluation behavior.
- In-deck interactives cover the regularization map, weight decay, augmentation/MixUp, dropout, and calibration.

## Handoff to the later uncertainty lecture

Students now know dropout as a training-time regularizer and know that ordinary evaluation disables it. A later uncertainty lecture can deliberately keep masks stochastic at inference and introduce MC Dropout, predictive uncertainty, calibration, and their failure modes.

Closing line:

> “During training, random masks make the shared weights cope with missing features. During evaluation, use the full network.”

## Primary sources

Zhang et al. (2017), rethinking generalization · Srivastava et al. (2014), dropout · Zhang et al. (2018), MixUp · Yun et al. (2019), CutMix · Cubuk et al. (2020), RandAugment · Szegedy et al. (2016), label smoothing · Müller et al. (2019), label-smoothing tradeoffs · Bishop (1995), training with noise · optional appendix: Loshchilov and Hutter (2019), AdamW.
