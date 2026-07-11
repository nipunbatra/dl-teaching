---
title: "Lecture 7: Generalization and Regularization in Deep Learning"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 7: Generalization and Regularization in Deep Learning

Core story:

\[
\boxed{
\text{Training loss is not the goal. Test performance is.}
}
\]

We now have networks that can train. Next problem:

\[
\text{How do we stop them from memorizing?}
\]

---

# Lecture 7 outline

## Part I — The generalization problem

### Slide 1 — Title

**Generalization and Regularization**

> How to make deep networks perform well on unseen data

---

### Slide 2 — Recall

Previous lectures gave us:

\[
f_\theta(x),\quad L(\theta),\quad \
abla_\theta L,\quad \theta_{t+1}
\]

Now ask:

\[
\boxed{
\text{Does low training loss imply good test loss?}
}
\]

Answer:

\[
\boxed{\text{No.}}
\]

---

### Slide 3 — Training vs test loss

Define:

\[
L_{\text{train}}(\theta)
=
\frac1{n_{\text{train}}}\sum_{i\in\text{train}}\ell_i(\theta)
\]

\[
L_{\text{test}}(\theta)
=
\mathbb E_{(x,y)\sim p_{\text{data}}}
[\ell(f_\theta(x),y)].
\]

But we only see finite data.

---

### Slide 4 — Generalization gap

\[
\text{gap}
=
L_{\text{test}}-L_{\text{train}}.
\]

Ideal:

\[
L_{\text{train}}\downarrow,\quad L_{\text{val}}\downarrow.
\]

Overfitting:

\[
L_{\text{train}}\downarrow,\quad L_{\text{val}}\uparrow.
\]

Visual: train/val loss curves.

---

### Slide 5 — Memorization example

Dataset:

\[
x_i\mapsto y_i
\]

Now randomly shuffle labels:

\[
x_i\mapsto y_{\pi(i)}.
\]

A large network can still drive training loss down.

Message:

> Capacity alone is not enough; we need inductive bias and validation.

---

## Part II — Data splits and model selection

### Slide 6 — Train/validation/test

Use:

- train: fit parameters;
- validation: choose hyperparameters/model;
- test: final unbiased evaluation.

\[
\text{Never tune on test repeatedly.}
\]

---

### Slide 7 — Early stopping as regularization

Stop at:

\[
t^\star=\arg\min_t L_{\text{val}}(\theta_t).
\]

Visual:

- train loss decreasing;
- validation loss U-shaped;
- selected checkpoint.

---

### Slide 8 — Worked example: early stopping

| Epoch | Train loss | Val loss |
|---:|---:|---:|
| 1 | 1.20 | 1.25 |
| 2 | 0.90 | 0.98 |
| 3 | 0.70 | 0.82 |
| 4 | 0.52 | 0.78 |
| 5 | 0.38 | 0.84 |
| 6 | 0.25 | 0.97 |

Choose epoch:

\[
t^\star=4.
\]

Not the last epoch.

---

### Slide 9 — Interactive 1: train/val curves

`train-val-overfit.html`

Controls:

- model size;
- noise level;
- dataset size;
- training epochs;
- early stopping patience.

Show:

- train loss;
- validation loss;
- selected checkpoint;
- generalization gap.

---

## Part III — Capacity, bias and variance

### Slide 10 — Underfitting vs overfitting

Underfitting:

\[
L_{\text{train}}\text{ high},\quad L_{\text{val}}\text{ high}.
\]

Overfitting:

\[
L_{\text{train}}\text{ low},\quad L_{\text{val}}\text{ high}.
\]

Good fit:

\[
L_{\text{train}}\text{ low},\quad L_{\text{val}}\text{ low}.
\]

Visual: three polynomial fits.

---

### Slide 11 — Bias-variance intuition

Prediction error decomposes conceptually into:

\[
\text{error}
=
\text{bias}^2
+
\text{variance}
+
\text{noise}.
\]

High bias:

\[
\text{model too simple}.
\]

High variance:

\[
\text{model too sensitive to dataset}.
\]

---

### Slide 12 — Worked polynomial example

Data:

\[
y=\sin(2\pi x)+\epsilon.
\]

Models:

\[
d=1,\quad d=5,\quad d=20.
\]

Show:

- \(d=1\): underfit;
- \(d=5\): good;
- \(d=20\): overfit.

This becomes the template for neural-network capacity.

---

### Slide 13 — Deep learning twist

Classical view:

\[
\text{larger model}\Rightarrow\text{more overfitting}.
\]

Modern DL:

- large models often generalize well;
- optimization, data, architecture and implicit regularization matter;
- still, validation discipline is non-negotiable.

Keep this nuanced.

---

## Part IV — Weight decay and \(L_2\)

### Slide 14 — Penalize large weights

Regularized objective:

\[
L_{\text{reg}}(\theta)
=
L_{\text{data}}(\theta)
+
\lambda\|\theta\|_2^2.
\]

From Lecture 1:

\[
L_2
\leftrightarrow
\text{Gaussian prior}.
\]

---

### Slide 15 — Gradient with \(L_2\)

\[
L_{\text{reg}}=L+\lambda\theta^\top\theta.
\]

Gradient:

\[
\
abla_\theta L_{\text{reg}}
=
\
abla_\theta L+2\lambda\theta.
\]

SGD update:

\[
\theta_{t+1}
=
\theta_t-\eta(\
abla L+2\lambda\theta_t)
\]

\[
=
(1-2\eta\lambda)\theta_t-\eta\
abla L.
\]

So parameters shrink every step.

---

### Slide 16 — Why “weight decay”?

The update:

\[
\theta_{t+1}
=
(1-\eta\lambda')\theta_t-\eta\
abla L
\]

contains multiplicative shrinkage:

\[
\theta_t\mapsto(1-\eta\lambda')\theta_t.
\]

This discourages large weights.

---

### Slide 17 — Worked example: one SGD step with weight decay

Let:

\[
\theta=5,\quad g=\
abla L=3,\quad \eta=0.1,\quad \lambda'=0.2.
\]

Without decay:

\[
\theta^+=5-0.1(3)=4.7.
\]

With decay:

\[
\theta^+=(1-0.1\cdot0.2)5-0.1(3)
\]

\[
=0.98\cdot5-0.3=4.6.
\]

Weight decay adds shrinkage.

---

### Slide 18 — AdamW: decoupled weight decay

For vanilla SGD, \(L_2\) regularization and weight decay are equivalent up to scaling. For adaptive optimizers such as Adam, Loshchilov and Hutter showed they are not equivalent and proposed decoupled weight decay, AdamW.

AdamW conceptually does:

\[
\theta
\leftarrow
\theta-\eta\cdot\operatorname{AdamStep}(g)
\]

then:

\[
\theta\leftarrow(1-\eta\lambda)\theta.
\]

---

### Slide 19 — When to use weight decay

Useful for:

- MLPs;
- CNNs;
- Transformers;
- large overparameterized models.

Usually do **not** decay:

- biases;
- normalization scale/shift parameters;
- sometimes embeddings depending on setup.

Practical message:

\[
\boxed{\text{Use AdamW, not Adam + naive }L_2,\text{ in modern setups.}}
\]

---

### Slide 20 — Interactive 2: weight decay

`weight-decay.html`

Controls:

- \(\lambda\);
- learning rate;
- optimizer: SGD / Adam / AdamW;
- polynomial degree.

Show:

- fitted curve;
- weight norm;
- train/val error;
- update decomposition.

---

## Part V — Dropout

### Slide 21 — Dropout idea

During training, randomly zero hidden units:

\[
m_j\sim\operatorname{Bernoulli}(p)
\]

\[
\tilde h=m\odot h.
\]

The key idea in dropout is randomly dropping units and their connections during training to reduce overfitting.

---

### Slide 22 — Inverted dropout

Use:

\[
\tilde h=\frac{m\odot h}{p}.
\]

Then:

\[
\mathbb E[\tilde h_j]
=
\mathbb E\left[\frac{m_jh_j}{p}\right]
=
h_j.
\]

So no scaling is needed at test time.

---

### Slide 23 — Worked dropout example

Let:

\[
h=[2,4,6,8],
\quad
p=0.5.
\]

Mask:

\[
m=[1,0,1,0].
\]

Inverted dropout:

\[
\tilde h=\frac{m\odot h}{0.5}
=
[4,0,12,0].
\]

Expected value:

\[
\mathbb E[\tilde h]=h.
\]

---

### Slide 24 — Dropout as model averaging intuition

Each mask selects a subnetwork.

Training samples many subnetworks.

At test time, the full network with scaled activations approximates averaging.

Do not overclaim exact Bayesian averaging.

---

### Slide 25 — Dropout caveats

Dropout:

- helps many MLPs;
- can help classifier heads;
- may be less useful with strong augmentation and normalization;
- can slow optimization;
- rate must be tuned.

Common rates:

\[
p=0.8\text{ to }0.5
\]

depending on layer and task.

---

### Slide 26 — Interactive 3: dropout

`dropout-regularization.html`

Controls:

- dropout keep probability \(p\);
- train/eval mode;
- model size;
- noise level.

Show:

- random masks;
- train/val curves;
- prediction variance;
- effective activation scaling.

---

## Part VI — Data augmentation

### Slide 27 — Regularize by changing the data

Instead of only modifying the model/objective, enlarge the effective training distribution.

\[
(x,y)\mapsto(T(x),y)
\]

where \(T\) preserves the label.

Examples:

- image crop;
- flip;
- colour jitter;
- noise;
- time shift;
- masking;
- text paraphrase.

---

### Slide 28 — Invariance

If label should not change under transformation \(T\):

\[
f_\theta(x)\approx f_\theta(T(x)).
\]

Data augmentation teaches invariance.

Visual:

```text
same dog image → crop/flip/color jitter → still dog
```

---

### Slide 29 — Bad augmentation

Not every transformation preserves the label.

Examples:

- flipping digits 6/9;
- time reversal in some signals;
- colour changes when colour is label-relevant;
- cropping out the object.

Message:

\[
\boxed{\text{augmentation encodes domain knowledge.}}
\]

---

### Slide 30 — Mixup

Sample two examples:

\[
(x_i,y_i),\quad(x_j,y_j).
\]

Draw:

\[
\lambda\sim\operatorname{Beta}(\alpha,\alpha).
\]

Create:

\[
\tilde x=\lambda x_i+(1-\lambda)x_j
\]

\[
\tilde y=\lambda y_i+(1-\lambda)y_j.
\]

Mixup trains on convex combinations of examples and labels and encourages approximately linear behaviour between training examples.

---

### Slide 31 — Worked mixup example

Suppose:

\[
x_A=\text{cat image},\quad y_A=[1,0]
\]

\[
x_B=\text{dog image},\quad y_B=[0,1]
\]

and:

\[
\lambda=0.7.
\]

Then:

\[
\tilde x=0.7x_A+0.3x_B
\]

\[
\tilde y=[0.7,0.3].
\]

Loss:

\[
-\sum_k \tilde y_k\log p_k.
\]

---

### Slide 32 — Interactive 4: augmentation and mixup

`augmentation-mixup.html`

Controls:

- augmentation strength;
- mixup \(\alpha\);
- dataset size;
- noise level.

Show:

- transformed samples;
- decision boundary;
- train/val gap;
- calibration/confidence.

---

## Part VII — Label smoothing

### Slide 33 — Hard one-hot labels

For \(K\) classes:

\[
y_k=
\begin{cases}
1,&k=y,\\
0,&k\
e y.
\end{cases}
\]

Cross-entropy:

\[
L=-\log p_y.
\]

This encourages:

\[
p_y\rightarrow1.
\]

---

### Slide 34 — Smooth labels

Label smoothing with parameter \(\epsilon\):

\[
y_k^{\text{smooth}}
=
\begin{cases}
1-\epsilon,&k=y,\\
\frac{\epsilon}{K-1},&k\
e y.
\end{cases}
\]

Alternative convention:

\[
y^{\text{smooth}}
=
(1-\epsilon)y+\epsilon\frac{\mathbf 1}{K}.
\]

---

### Slide 35 — Worked label smoothing example

Let:

\[
K=5,\quad \epsilon=0.1.
\]

Correct class:

\[
1-\epsilon=0.9.
\]

Other classes:

\[
\frac{0.1}{4}=0.025.
\]

So target:

\[
[0.025,0.025,0.9,0.025,0.025].
\]

---

### Slide 36 — Why label smoothing helps

It discourages extreme confidence:

\[
p_y=1,\quad p_{k\
e y}=0.
\]

Can improve:

- calibration;
- robustness to noisy labels;
- generalization.

But can hurt when exact confidence or distillation targets matter.

---

### Slide 37 — Interactive 5: label smoothing

`label-smoothing.html`

Controls:

- number of classes;
- \(\epsilon\);
- logits;
- true class.

Show:

- target distribution;
- cross-entropy;
- gradient:
  \[
  p-y^{\text{smooth}}.
  \]
- confidence penalty effect.

---

## Part VIII — Regularization by noise and constraints

### Slide 38 — Noise as regularization

Sources of training noise:

- SGD minibatches;
- dropout masks;
- data augmentation;
- stochastic depth;
- label noise;
- input noise.

Noise prevents brittle dependence on exact training examples.

---

### Slide 39 — Explicit vs implicit regularization

Explicit:

\[
+\lambda\|\theta\|^2
\]

dropout, augmentation, label smoothing.

Implicit:

- SGD noise;
- architecture bias;
- optimizer bias;
- early stopping;
- normalization;
- initialization.

Message:

\[
\boxed{\text{Regularization is not only a penalty term.}}
\]

---

### Slide 40 — Regularization changes the learned function

It is not merely about reducing parameters.

Two models may both fit training data:

\[
L_{\text{train}}\approx0.
\]

Regularization chooses the one with:

- smoother boundary;
- smaller norm;
- more invariance;
- less confidence;
- better validation loss.

Visual: two decision boundaries fitting same points.

---

## Part IX — Practical workflow

### Slide 41 — Debug before regularizing

If both losses are high:

\[
L_{\text{train}}\text{ high},\quad L_{\text{val}}\text{ high}
\]

you are underfitting.

Do **not** add more regularization.

Instead:

- bigger model;
- train longer;
- better optimizer/LR;
- reduce excessive augmentation;
- check labels/data.

---

### Slide 42 — If train low, val high

Overfitting signs:

\[
L_{\text{train}}\downarrow,\quad L_{\text{val}}\uparrow.
\]

Try:

1. more data;
2. augmentation;
3. weight decay;
4. dropout;
5. early stopping;
6. smaller model;
7. label smoothing.

---

### Slide 43 — Regularization dashboard

Track:

- train loss;
- validation loss;
- train accuracy;
- validation accuracy;
- weight norm:
  \[
  \|\theta\|_2
  \]
- prediction confidence:
  \[
  \max_k p_k
  \]
- calibration error if relevant.

---

### Slide 44 — Worked diagnosis

Scenario A:

\[
\text{train acc}=65\%,\quad \text{val acc}=63\%.
\]

Diagnosis:

\[
\text{underfit}
\]

Action:

> increase capacity / train better.

Scenario B:

\[
\text{train acc}=99\%,\quad \text{val acc}=72\%.
\]

Diagnosis:

\[
\text{overfit}
\]

Action:

> regularize / augment / early stop.

---

### Slide 45 — Practical defaults

For image classification:

\[
\text{augmentation + weight decay + early stopping/checkpointing}
\]

For MLP tabular:

\[
\text{weight decay + dropout + early stopping}
\]

For Transformers:

\[
\text{AdamW + dropout + label smoothing sometimes + data augmentation/task-specific noise}
\]

For small data:

\[
\text{transfer learning + strong validation discipline}
\]

---

## Part X — Summary

### Slide 46 — One table

| Method | What it controls |
|---|---|
| early stopping | training time/effective complexity |
| weight decay | parameter norm |
| dropout | co-adaptation / noise robustness |
| augmentation | invariance |
| mixup | linearity between examples |
| label smoothing | overconfidence |
| validation split | model selection |

---

### Slide 47 — Final mental model

\[
\boxed{
\text{Optimization reduces training loss.}
}
\]

\[
\boxed{
\text{Regularization improves unseen-data performance.}
}
\]

\[
\boxed{
\text{Validation tells us whether regularization is helping.}
}
\]

---

### Slide 48 — Next lecture

**CNNs**

Why fully connected networks are inefficient for images.

Topics:

- locality;
- convolution;
- padding/stride;
- pooling;
- channels;
- CNN architectures.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Generalization problem | 1–5 | 8 min |
| Splits and early stopping | 6–9 | 8 min |
| Bias/variance/capacity | 10–13 | 8 min |
| Weight decay/AdamW | 14–20 | 13 min |
| Dropout | 21–26 | 10 min |
| Augmentation/mixup | 27–32 | 12 min |
| Label smoothing | 33–37 | 8 min |
| Regularization concepts | 38–40 | 5 min |
| Practical workflow | 41–45 | 6 min |
| Summary | 46–48 | 2 min |

---

# Main diagrams

1. Train/validation loss curves.
2. Underfit/good fit/overfit polynomial curves.
3. Generalization gap visual.
4. Weight decay shrinkage vector.
5. Adam vs AdamW update decomposition.
6. Dropout mask over hidden layer.
7. Data augmentation invariance examples.
8. Mixup interpolation diagram.
9. Label smoothing target distribution.
10. Decision boundary with and without regularization.

---

# Interactives

1. `train-val-overfit.html`  
   Model size, epochs, early stopping, generalization gap.

2. `weight-decay.html`  
   Polynomial fit or 2D classifier with \(\lambda\), SGD/Adam/AdamW.

3. `dropout-regularization.html`  
   Masks, train/eval mode, prediction variance.

4. `augmentation-mixup.html`  
   Augmentation strength and mixup \(\alpha\), decision boundary.

5. `label-smoothing.html`  
   Soft targets, CE loss, gradient \(p-y\).

6. `regularization-dashboard.html`  
   Combine weight decay, dropout, augmentation and early stopping.

---

# Notebooks

1. `01_overfitting_polynomial.ipynb`
2. `02_train_val_early_stopping.ipynb`
3. `03_weight_decay_from_scratch.ipynb`
4. `04_adam_vs_adamw.ipynb`
5. `05_dropout_numpy_pytorch.ipynb`
6. `06_augmentation_mixup.ipynb`
7. `07_label_smoothing.ipynb`
8. `08_regularization_ablation.ipynb`
