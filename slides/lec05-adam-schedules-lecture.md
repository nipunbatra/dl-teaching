---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Adam, AdamW & LR Schedules

## Lecture 5 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain why **per-parameter** LR is helpful (sparse vs dense gradients).
2. Derive **Adam** as momentum + RMSProp + bias correction.
3. Compute **bias-correction** at small $t$ and show why it matters.
4. Distinguish **Adam vs AdamW** and pick AdamW for regularized training.
5. Pick a **schedule** (constant / step / cosine / warmup+cosine) per use-case.
6. Explain **why warmup** is essential for Transformers.

---

# Recap · where we left off

Lecture 4 · **momentum** = EMA of past gradients. Damps ravine oscillation and speeds training.

But momentum still uses a **single learning rate** for all parameters.

<div class="paper">

Today maps to **UDL Ch 6** (Adam) and **Ch 7** (gradients + initialization revisited).

</div>

**Q.** Is that always the right thing?

---

# Not always — here's why

Imagine training a model where:

- word-embedding parameters are updated by rare tokens → gradients are **large and sparse**
- hidden-layer weights are updated every step → gradients are **small but constant**

A single LR that is right for one is wrong for the other.

<div class="keypoint">

Today · **per-parameter adaptive learning rates** — AdaGrad → RMSProp → Adam → AdamW — plus the schedule we wrap around them.

</div>

---

# Four questions

1. How do we get a **per-parameter** learning rate?
2. What is **Adam** actually doing, piece by piece?
3. Why is L2 "broken" inside Adam, and how does **AdamW** fix it?
4. What **schedule** should you use, and why do Transformers need warmup?

---

<!-- _class: section-divider -->

### PART 1

# The family tree

SGD → AdaGrad → RMSProp → Adam

---

# The lineage

![w:920px](figures/lec05/svg/optimizer_family_tree.svg)

---

# AdaGrad — the first per-parameter LR (2011)

<div class="math-box">

$$G_t = G_{t-1} + g_t^2, \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}}\, g_t$$

</div>

- Parameters that got large gradients → denominator grows → effective LR shrinks.
- Parameters that got small gradients → effective LR stays large.

**Q.** What goes wrong with $G_t = G_{t-1} + g_t^2$ over many steps?

---

# AdaGrad's problem · LR decays to zero

![w:920px](figures/lec05/svg/adagrad_decay.svg)

---

# Adam · Momentum + RMSProp

![w:920px](figures/lec05/svg/adam_components.svg)

---

# Trajectories · SGD vs momentum vs Adam

![w:920px](figures/lec05/svg/optimizer_trajectories.svg)

---

# Adam · the full update

<div class="math-box">

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t   \qquad \text{(1st moment)}$$

$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2 \qquad \text{(2nd moment)}$$

$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}  \qquad \text{(bias corr.)}$$

$$\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Defaults · $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}, \eta = 10^{-3}$.

</div>

---

# Adam · worked example · 3 steps on a single parameter

Suppose · gradients $g_1, g_2, g_3 = 2.0, 1.0, 1.5$. Defaults; $\theta_0 = 0$.

<div class="math-box">

| step | $m_t$ | $\hat m_t$ | $v_t$ | $\hat v_t$ | update |
|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | $0.9 \cdot 0 + 0.1 \cdot 2.0 = 0.20$ | $0.20 / 0.1 = 2.00$ | $0.001 \cdot 4 = 0.004$ | $0.004 / 0.001 = 4.0$ | $-10^{-3} \cdot 2 / 2 = -0.001$ |
| 2 | $0.9 \cdot 0.20 + 0.1 \cdot 1.0 = 0.28$ | $0.28 / 0.19 = 1.47$ | $0.999 \cdot 0.004 + 0.001 \cdot 1 = 0.005$ | $0.005 / 0.001999 \approx 2.5$ | $-10^{-3} \cdot 1.47 / 1.58 \approx -0.00093$ |
| 3 | $0.9 \cdot 0.28 + 0.1 \cdot 1.5 = 0.402$ | $0.402 / 0.271 = 1.48$ | $0.999 \cdot 0.005 + 0.001 \cdot 2.25 = 0.0072$ | $\approx 2.4$ | $\approx -0.00095$ |

</div>

Note · the per-parameter update size is **roughly constant ($\sim 10^{-3}$)** even though the gradients change. That's Adam's per-parameter adaptive scaling at work.

---

<!-- _class: section-divider -->

### PART 2

# The bias-correction detail

Why we divide by $(1 - \beta^t)$

---

# What goes wrong at $t = 1$?

Initialize $m_0 = 0, v_0 = 0$. At step $t = 1$:

$$m_1 = \beta_1 \cdot 0 + (1 - \beta_1)\, g_1 = 0.1\, g_1$$

The EMA is **10× smaller** than the true gradient — because it started from zero.

---

# The correction in one picture

![w:920px](figures/lec05/svg/adam_bias_correction.svg)

---

# ⚠️ optional · The fix, derived

$E[g_t]$ is (roughly) stationary $\mu$. Then:

$$E[m_t] = (1 - \beta_1)\sum_{k=1}^{t} \beta_1^{t-k}\, E[g_k] = \mu\, (1 - \beta_1^t)$$

So the unbiased estimator is:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

Same idea for $v_t$ and $\beta_2$.

---

# Worked example · bias correction in action

<div class="math-box">

Suppose the true gradient is $g = 1.0$ at every step. With $\beta_1 = 0.9$, starting $m_0 = 0$:

| step $t$ | $m_t = 0.9 m_{t-1} + 0.1 g$ | $\hat{m}_t = m_t / (1 - 0.9^t)$ |
|:-:|:-:|:-:|
| 1 | 0.100 | **1.000** |
| 2 | 0.190 | **1.000** |
| 3 | 0.271 | **1.000** |
| 10 | 0.651 | **1.000** |
| 100 | 1.000 | 1.000 |

</div>

Without correction, the first step would use $m_1 = 0.1$ — ten times too small. Adam's step would therefore be **10× smaller than the optimal at t=1**, wasting the early training phase. The $(1 - \beta_1^t)$ denominator fixes it exactly.

---

# When does bias correction matter?

For $\beta_1 = 0.9$:
- $t = 1$: factor $\frac{1}{1 - 0.9} = 10$ ← huge
- $t = 10$: $\frac{1}{1 - 0.9^{10}} \approx 1.54$
- $t = 100$: $\approx 1.000026$ ← negligible

Bias correction is a first-few-steps phenomenon. It keeps early steps the right magnitude; after that it fades to the identity.

---

<!-- _class: section-divider -->

### PART 3

# Adam → AdamW

The decoupled-weight-decay fix

---

# L2 · quick recap from ES 654

L2 regularization adds $\frac{\lambda}{2}\|\theta\|^2$ to the loss. Gradient contribution: $\lambda\, \theta$.

In plain SGD, this equals *weight decay* — each step shrinks weights by a factor of $(1 - \eta\lambda)$:

$$\theta_t = \theta_{t-1} - \eta\,\nabla \mathcal{L} - \eta \lambda\, \theta_{t-1}$$

For SGD these are the same. **Not so for Adam.**

---

# AdamW · the fix

![w:920px](figures/lec05/svg/adamw_vs_adam.svg)

---

# Adam vs AdamW · one-step worked numeric

Setup · $\theta = 1.0$, $g = 0.5$, $\lambda = 0.1$ (weight decay), $\eta = 10^{-3}$, $\hat v = 4.0$ (from RMSProp).

<div class="math-box">

**Adam (L2 in gradient)**
- Effective gradient · $g + \lambda \theta = 0.5 + 0.1 = 0.6$
- Update · $-\eta \cdot 0.6 / \sqrt{4} = -0.0003$
- Note · the "regularization" $\lambda \theta$ got divided by $\sqrt{\hat v}$ · weakened on high-gradient params!

**AdamW (decoupled)**
- Adaptive update · $-\eta \cdot 0.5 / \sqrt{4} = -0.00025$
- Plus uniform decay · $- \eta \lambda \theta = -0.0001$
- Total · $-0.00035$

</div>

In AdamW, weight decay is **uniform** for every parameter. In Adam, parameters with large past gradients are decayed less. AdamW's behavior matches the regularization theory · use it.

---

# AdamW in PyTorch · one line

```python
# the right default for almost everything in 2026
opt = torch.optim.AdamW(model.parameters(),
                        lr=3e-4,
                        betas=(0.9, 0.999),
                        weight_decay=0.1)   # typical for LLMs
```

<div class="realworld">

LLMs: `weight_decay=0.1`. Fine-tuning: `0.01–0.05`. Vision fine-tune: `0.001–0.01`.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Learning-rate schedules

Why one learning rate isn't enough over a full run

---

# LR schedules · four common shapes

![w:920px](figures/lec05/svg/lr_schedule_shapes.svg)

---

# Four common schedules

![w:920px](figures/lec05/svg/lr_schedules.svg)

<div class="realworld">

▶ Interactive: sliders for peak LR and warmup — [lr-schedule-visualizer](https://nipunbatra.github.io/interactive-articles/lr-schedule-visualizer/).

</div>

---

# Schedules in PyTorch

```python
# Step decay
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[60, 120], gamma=0.1)

# Cosine annealing
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

# Warmup + cosine — the 2026 default for Transformers
from torch.optim.lr_scheduler import LambdaLR
def lr_lambda(step):
    if step < warmup: return step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))
sched = LambdaLR(opt, lr_lambda)
```

---

# Why Transformers need warmup

![w:920px](figures/lec05/svg/warmup_necessity.svg)

---

# Why Transformers need warmup · the explanation

Adam's $\hat{v}_t$ is tiny and noisy at the start. Dividing by $\sqrt{\hat{v}_t}$ amplifies step sizes wildly.

Meanwhile, random-init Transformer weights produce **peaky** attention distributions — large early gradients in a few heads.

Combined: huge, unstable first steps → divergence.

<div class="keypoint">

**Warmup** — linearly ramp LR from 0 to its target over ~1–10% of training. Tames early instability; after warmup, use cosine decay.

</div>

---

<!-- _class: section-divider -->

### PART 5

# What to actually use

---

# Warmup · typical schedule

<div class="math-box">

Typical Transformer schedule:

- **Warmup** · linearly ramp lr 0 → target over first 2000 steps (or 1-10% of total).
- **Plateau** · hold at target lr briefly.
- **Cosine decay** · smooth drop from target to ~1/10 of target over the rest.
- **Final** · ~10% of training at minimum lr to squeeze last gains.

</div>

```python
def lr_lambda(step, total, warmup):
    if step < warmup:
        return step / warmup       # linear warmup 0 → 1
    progress = (step - warmup) / (total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))  # cosine
```

5 lines. Ubiquitous.

---

# Defaults that work

| Model / regime | Optimizer | LR | Schedule |
|----------------|-----------|-----|----------|
| CNN from scratch | **SGD + momentum + Nesterov** | 0.1 | step decay |
| CNN fine-tune | AdamW | 1e-4 | cosine |
| Transformer pre-train | **AdamW** (β₂ = 0.95) | 3e-4 | **warmup + cosine** |
| LoRA fine-tune of LLM | AdamW | 1e-4 to 3e-4 | cosine |
| Debugging a new idea | AdamW | 3e-4 | constant |

<div class="realworld">

`lr = 3e-4` is not magic — it's the number to use when you don't want to think. For a real run, do the **LR finder** (Lecture 3).

</div>

---

# Gradient clipping · cheap insurance

![w:920px](figures/lec05/svg/gradient_clipping.svg)

---

# Common mistakes

<div class="warning">

**Leaving `weight_decay` at 0 for AdamW.** You get plain Adam with no regularization. Surprisingly common.

**No LR schedule for an LLM.** Training plateaus early and you blame the architecture.

**Warmup of 10 steps for a 100k-step run.** Far too short. Warmup = 1–10% of total.

**`lr = 3e-4` for SGD.** That's the Adam default. SGD usually wants `lr = 0.01` to `0.1`.

</div>

---

<!-- _class: summary-slide -->

# Lecture 5 — summary

- **AdaGrad** gave per-parameter LR — but $G_t$ never forgets, so LRs decay to zero.
- **RMSProp** fixed that with EMA of $g_t^2$.
- **Adam = momentum + RMSProp + bias correction.** Robust first-try optimizer.
- **Bias correction** matters for the first ~10–100 steps; $\hat{m}_t = m_t / (1 - \beta^t)$.
- **AdamW** decouples weight decay from adaptive scaling. Use it, not Adam+L2.
- **Schedules** — cosine is the clean default; **warmup + cosine** is the Transformer default.
- **Gradient clipping at 1.0** — cheap insurance.

### Read before Lecture 6

**Prince** — Ch 6 §6.7 (Adam), Ch 7. Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

**Regularization I** — bias-variance in the overparameterized regime, double descent, weight decay as prior, early stopping, data augmentation, Mixup, label smoothing.

<div class="notebook">

**Notebook 5** · `05-adam-schedules.ipynb` — implement Adam and AdamW from scratch; sweep step-decay vs cosine on CIFAR-10.

</div>
