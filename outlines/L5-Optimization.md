---
title: "Lecture 5: Optimization for Deep Learning"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 5: Optimization for Deep Learning  
## SGD, Momentum, Adam, Learning Rates

Core story:

\[
\boxed{
\text{Backprop gives gradients. Optimization decides how to use them.}
}
\]

---

## Big narrative

\[
\
abla_\theta L
\rightarrow
\text{gradient descent}
\rightarrow
\text{SGD as unbiased noisy estimate}
\rightarrow
\text{momentum smooths noise}
\rightarrow
\text{Adam adapts step sizes}
\]

SGD comes from the stochastic approximation tradition of Robbins–Monro; Adam uses adaptive first and second moment estimates.

---

# Slide-level outline

## Part I — From backprop to optimization

### 1. Title  
**Optimization for Deep Learning**

### 2. Recall  
Backprop gives:

\[
g_t=\
abla_\theta L(\theta_t)
\]

Optimization chooses:

\[
\theta_{t+1}=\text{update}(\theta_t,g_t)
\]

### 3. Loss landscape picture  
Diagram: surface plot + contour plot.

Show:

- current point;
- gradient arrow uphill;
- negative gradient downhill.

### 4. Basic gradient descent

\[
\theta_{t+1}
=
\theta_t-\eta\
abla_\theta L(\theta_t)
\]

where \(\eta\) is the learning rate.

### 5. Worked example: 1D quadratic

\[
L(\theta)=(\theta-3)^2
\]

\[
\
abla_\theta L=2(\theta-3)
\]

Update:

\[
\theta_{t+1}
=
\theta_t-2\eta(\theta_t-3)
\]

Use \(\theta_0=0\).

For \(\eta=0.1\):

\[
\theta_1=0.6,\quad
\theta_2=1.08,\quad
\theta_3=1.464.
\]

### 6. Learning rate matters

For same quadratic:

- small \(\eta\): slow;
- good \(\eta\): fast;
- large \(\eta\): oscillation;
- too large: divergence.

Diagram: four trajectories on same 1D curve.

---

## Part II — Batch GD versus SGD

### 7. Empirical risk

\[
L(\theta)=\frac1n\sum_{i=1}^n \ell_i(\theta)
\]

Full gradient:

\[
\
abla L(\theta)
=
\frac1n\sum_{i=1}^n \
abla \ell_i(\theta)
\]

### 8. Full-batch gradient descent

\[
g_t=\frac1n\sum_i\
abla\ell_i(\theta_t)
\]

Pros:

- stable;
- exact empirical gradient.

Cons:

- expensive for large \(n\);
- one update per full dataset pass.

### 9. Stochastic gradient descent

Sample one example \(I\sim\operatorname{Uniform}\{1,\ldots,n\}\).

\[
g_t=\
abla\ell_I(\theta_t)
\]

Update:

\[
\theta_{t+1}=\theta_t-\eta g_t.
\]

### 10. SGD gradient is unbiased

\[
\mathbb E_I[g_t]
=
\mathbb E_I[\
abla\ell_I(\theta_t)]
\]

\[
=
\frac1n\sum_{i=1}^n\
abla\ell_i(\theta_t)
=
\
abla L(\theta_t).
\]

So:

\[
\boxed{\mathbb E[g_t]=\
abla L(\theta_t)}
\]

### 11. But SGD is noisy

\[
g_t=\
abla L(\theta_t)+\epsilon_t
\]

where:

\[
\mathbb E[\epsilon_t]=0.
\]

Variance:

\[
\operatorname{Var}(g_t)
=
\mathbb E\|g_t-\
abla L\|^2.
\]

Diagram: noisy gradient arrows around true gradient.

### 12. Minibatch SGD

Sample batch:

\[
\mathcal B_t\subset\{1,\ldots,n\}
\]

\[
g_t=
\frac1{|\mathcal B_t|}
\sum_{i\in\mathcal B_t}
\
abla\ell_i(\theta_t)
\]

Still unbiased:

\[
\mathbb E[g_t]=\
abla L(\theta_t).
\]

Variance decreases roughly with batch size:

\[
\operatorname{Var}(g_{\mathcal B})
\approx
\frac{1}{B}\operatorname{Var}(g_i)
\]

assuming independent samples.

### 13. Batch size tradeoff

Small batch:

- noisy;
- cheap updates;
- often better exploration.

Large batch:

- stable;
- hardware efficient;
- fewer updates per epoch.

### 14. Diagram: batch size and noise

Show contour plot:

- full batch: smooth path;
- mini-batch: jittery but directed path;
- tiny batch: very noisy path.

### 15. Interactive 1: SGD noise

`sgd-noise.html`

Controls:

- batch size;
- learning rate;
- dataset size;
- noise level.

Show:

- true gradient;
- sampled minibatch gradient;
- trajectory;
- gradient variance.

---

## Part III — Momentum

### 16. Motivation: ravines

Use elongated quadratic:

\[
L(x,y)=x^2+100y^2.
\]

Gradient descent zigzags because curvature differs by direction.

Diagram: narrow valley contour.

### 17. Momentum idea

Instead of using only current gradient, keep velocity:

\[
v_t=\beta v_{t-1}+g_t
\]

\[
\theta_{t+1}=\theta_t-\eta v_t.
\]

Alternative common convention:

\[
v_t=\beta v_{t-1}-\eta g_t
\]

\[
\theta_{t+1}=\theta_t+v_t.
\]

### 18. Exponentially weighted average

Unroll:

\[
v_t
=
g_t+\beta g_{t-1}+\beta^2g_{t-2}+\cdots
\]

So momentum averages recent gradients.

\[
\boxed{\text{momentum remembers direction}}
\]

### 19. Why momentum helps

Momentum:

- damps oscillations across valley;
- accelerates along consistent direction;
- smooths gradient noise.

Diagram: GD zigzag path versus momentum path.

### 20. Worked example: noisy gradients

Suppose 1D gradients:

\[
g_1=10,\quad g_2=8,\quad g_3=11
\]

with:

\[
\beta=0.9,\quad v_0=0.
\]

Then:

\[
v_1=10
\]

\[
v_2=0.9(10)+8=17
\]

\[
v_3=0.9(17)+11=26.3
\]

If using normalized EMA:

\[
m_t=\beta m_{t-1}+(1-\beta)g_t
\]

then:

\[
m_1=1,\quad m_2=1.7,\quad m_3=2.63.
\]

Explain convention difference.

### 21. Momentum hyperparameter

Typical:

\[
\beta=0.9
\]

Effective memory length:

\[
\frac1{1-\beta}
\]

So:

\[
\beta=0.9\Rightarrow \text{about }10\text{ steps}
\]

\[
\beta=0.99\Rightarrow \text{about }100\text{ steps}
\]

### 22. Interactive 2: momentum in ravine

`momentum-ravine.html`

Controls:

- \(\eta\);
- \(\beta\);
- curvature ratio;
- start point.

Show:

- GD path;
- momentum path;
- velocity vector;
- loss curve.

---

## Part IV — Adaptive methods: RMSProp idea to Adam

### 23. Different parameters need different step sizes

Some coordinates have large gradients; others small.

Problem:

\[
\theta_{t+1,j}
=
\theta_{t,j}-\eta g_{t,j}
\]

uses same \(\eta\) for all coordinates.

### 24. Adaptive step intuition

Use smaller steps for coordinates with consistently large gradients.

Use larger effective steps for coordinates with consistently small gradients.

Generic form:

\[
\theta_{t+1,j}
=
\theta_{t,j}
-
\frac{\eta}{\sqrt{s_{t,j}}+\epsilon}
g_{t,j}
\]

### 25. Squared-gradient accumulator

\[
s_t=\rho s_{t-1}+(1-\rho)g_t^2
\]

elementwise.

Then:

\[
\theta_{t+1}
=
\theta_t
-
\eta
\frac{g_t}{\sqrt{s_t}+\epsilon}.
\]

This is RMSProp-style intuition.

### 26. Why divide by RMS gradient?

If a coordinate has large historical gradient magnitude:

\[
s_{t,j}\text{ large}
\Rightarrow
\text{smaller step}.
\]

If small:

\[
s_{t,j}\text{ small}
\Rightarrow
\text{larger step}.
\]

Diagram: coordinate-wise rescaling.

---

## Part V — Adam

### 27. Adam combines momentum and adaptivity

Adam maintains:

First moment:

\[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
\]

Second moment:

\[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
\]

Adam was proposed as an efficient first-order stochastic optimizer using adaptive estimates of lower-order moments.

### 28. Bias in moving averages

Initialize:

\[
m_0=0.
\]

Then:

\[
m_1=(1-\beta_1)g_1.
\]

So early \(m_t\) is biased toward zero.

Similarly \(v_t\) is biased toward zero.

### 29. Bias correction

Adam uses:

\[
\hat m_t=\frac{m_t}{1-\beta_1^t}
\]

\[
\hat v_t=\frac{v_t}{1-\beta_2^t}
\]

Update:

\[
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
\]

### 30. Worked Adam example

Let scalar gradients:

\[
g_1=4,\quad g_2=6.
\]

Use:

\[
\beta_1=0.9,\quad \beta_2=0.999,\quad \eta=0.001,\quad \epsilon\approx0.
\]

Step 1:

\[
m_1=0.1\cdot4=0.4
\]

\[
v_1=0.001\cdot16=0.016
\]

Bias correction:

\[
\hat m_1=\frac{0.4}{1-0.9}=4
\]

\[
\hat v_1=\frac{0.016}{1-0.999}=16
\]

Update:

\[
\Delta\theta_1
=
-0.001\frac{4}{\sqrt{16}}
=
-0.001.
\]

Step 2:

\[
m_2=0.9(0.4)+0.1(6)=0.96
\]

\[
v_2=0.999(0.016)+0.001(36)=0.051984
\]

\[
\hat m_2=\frac{0.96}{1-0.9^2}=5.0526
\]

\[
\hat v_2=\frac{0.051984}{1-0.999^2}\approx26.005
\]

\[
\Delta\theta_2
\approx
-0.001\frac{5.0526}{\sqrt{26.005}}
\approx
-0.000991.
\]

Message:

> Adam normalizes by recent gradient scale.

### 31. Adam default hyperparameters

Common defaults from the Adam paper:

\[
\beta_1=0.9,\quad
\beta_2=0.999,\quad
\epsilon=10^{-8}.
\]

### 32. Adam intuition

\[
\hat m_t
\approx
\text{smoothed gradient direction}
\]

\[
\sqrt{\hat v_t}
\approx
\text{smoothed gradient magnitude}
\]

So:

\[
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
\]

is a normalized adaptive update.

### 33. Interactive 3: Adam mechanics

`adam-step.html`

Controls:

- gradient sequence;
- \(\beta_1,\beta_2\);
- \(\eta\);
- with/without bias correction.

Show:

- \(g_t\);
- \(m_t\);
- \(v_t\);
- \(\hat m_t\);
- \(\hat v_t\);
- update size.

---

## Part VI — Comparing optimizers

### 34. Same loss, different optimizers

Use:

\[
L(x,y)=x^2+50y^2
\]

Show paths:

- GD;
- SGD;
- Momentum;
- Adam.

### 35. Summary table

| Optimizer | Update memory | Adaptive scale | Strength |
|---|---:|---:|---|
| GD | no | no | stable full gradient |
| SGD | no | no | cheap noisy updates |
| Momentum | first moment | no | smoother, faster valleys |
| Adam | first + second moments | yes | robust default |

### 36. When to use what?

Practical default:

- Adam/AdamW for many modern DL workloads;
- SGD+momentum still common when final generalization is critical, especially classical vision training;
- learning-rate schedule often matters as much as optimizer.

Keep this as pragmatic, not dogma.

### 37. AdamW note

If discussing regularization:

Adam with \(L_2\) penalty is not exactly the same as decoupled weight decay.

AdamW decouples weight decay from gradient update.

Optional slide only; save for regularization/training tricks lecture.

---

## Part VII — Learning-rate schedules

### 38. Fixed learning rate is rarely ideal

Early training:

\[
\text{large steps useful}
\]

Late training:

\[
\text{small steps useful}
\]

### 39. Common schedules

Show curves:

- step decay;
- exponential decay;
- cosine decay;
- warmup + cosine;
- one-cycle.

### 40. Warmup

Useful when early gradients/normalization/statistics are unstable.

Schedule:

\[
\eta_t
=
\eta_{\max}\frac{t}{T_{\text{warmup}}}
\]

for:

\[
t\le T_{\text{warmup}}.
\]

Then decay.

### 41. Interactive 4: learning rate schedule

`lr-schedules.html`

Controls:

- schedule type;
- max LR;
- warmup steps;
- decay rate.

Show:

- LR curve;
- optimizer trajectory on contour plot.

---

## Part VIII — Optimization pathologies

### 42. Noisy gradients

SGD has useful noise, but too much noise prevents convergence.

Diagram: bouncing around minimum.

### 43. Vanishing/exploding gradients

From earlier:

\[
\prod_\ell J_\ell
\]

can shrink or explode.

Optimization interacts with:

- activation;
- initialization;
- normalization;
- residual connections.

### 44. Saddle points

At saddle:

\[
\
abla L=0
\]

but Hessian has mixed eigenvalues.

SGD noise can help escape some saddles.

### 45. Sharp versus flat minima

Visual: two minima.

Mention carefully:

- flat minima are often associated with robustness/generalization;
- sharpness definitions are subtle and scale-dependent.

Optional.

---

## Part IX — End-to-end training loop

### 46. Full PyTorch loop

```python
for x, y in loader:
    optimizer.zero_grad()

    logits = model(x)
    loss = loss_fn(logits, y)

    loss.backward()
    optimizer.step()
```

Map each line:

\[
\text{forward}
\rightarrow
\text{loss}
\rightarrow
\text{backprop}
\rightarrow
\text{update}
\]

### 47. SGD implementation

```python
with torch.no_grad():
    for p in model.parameters():
        p -= lr * p.grad
```

### 48. Momentum implementation

```python
v = beta * v + grad
param -= lr * v
```

### 49. Adam implementation sketch

```python
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad**2

mhat = m / (1 - beta1**t)
vhat = v / (1 - beta2**t)

param -= lr * mhat / (sqrt(vhat) + eps)
```

---

## Part X — Summary

### 50. One-slide summary

\[
\boxed{\text{GD: use full gradient}}
\]

\[
\boxed{\text{SGD: use unbiased noisy gradient estimate}}
\]

\[
\boxed{\text{Momentum: average gradients over time}}
\]

\[
\boxed{\text{Adam: momentum + adaptive coordinate scaling}}
\]

\[
\boxed{\text{LR schedule: controls step size over training}}
\]

### 51. Final mental model

Backprop answers:

\[
\text{What direction reduces loss?}
\]

Optimization answers:

\[
\text{How far, how smoothly, and how adaptively should we move?}
\]

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Motivation + GD | 1–6 | 9 min |
| SGD + unbiased estimator | 7–15 | 16 min |
| Momentum | 16–22 | 13 min |
| Adaptive methods + Adam | 23–33 | 18 min |
| Comparison | 34–37 | 7 min |
| LR schedules | 38–41 | 7 min |
| Pathologies | 42–45 | 5 min |
| Code loop + summary | 46–51 | 5 min |

---

# Main diagrams to make

1. **Surface + contour + gradient arrow**
2. **Learning-rate trajectories on 1D quadratic**
3. **Full-batch vs minibatch noisy gradient arrows**
4. **GD zigzag in ravine**
5. **Momentum smoothing path**
6. **Adam as first moment + second moment normalization**
7. **Optimizer comparison on same contour**
8. **LR schedule curves**
9. **Sharp versus flat minima**

---

# Interactives

1. `gd-quadratic.html`  
   1D and 2D GD, learning-rate effects.

2. `sgd-noise.html`  
   Batch size, unbiased gradient estimate, noise variance.

3. `momentum-ravine.html`  
   Momentum versus GD in elongated quadratic.

4. `adam-step.html`  
   Show \(m_t,v_t,\hat m_t,\hat v_t\), update magnitude.

5. `optimizer-race.html`  
   GD, SGD, Momentum, Adam on same landscape.

6. `lr-schedules.html`  
   Warmup, step decay, cosine, exponential.

---

# Notebooks

1. `01_gd_quadratic.ipynb`
2. `02_sgd_unbiased_estimator.ipynb`
3. `03_minibatch_variance.ipynb`
4. `04_momentum_from_scratch.ipynb`
5. `05_adam_from_scratch.ipynb`
6. `06_optimizer_comparison_pytorch.ipynb`
7. `07_lr_schedules.ipynb`

Make **Slide 10–12** mathematically strong. That will make SGD feel principled rather than “just faster.”
