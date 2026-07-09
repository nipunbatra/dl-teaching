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

# Not always — one global LR is a compromise

Loss surfaces are **lopsided**. In some directions the loss is a **steep canyon**; in others a nearly-**flat plain**.

- **Steep direction** · gradients are large → a big step *overshoots* and bounces.
- **Flat direction** · gradients are tiny → a small step *crawls* and wastes time.

One global $\eta$ can't win both — big enough for the plain overshoots the canyon; small enough for the canyon crawls on the plain. **SGD uses the same $\eta$ everywhere and ignores this.**

*(Same story with sparse vs dense parameters: rare-token embeddings get large, occasional gradients; hidden weights get small, constant ones.)*

<div class="keypoint">

The fix · give **every parameter its own step size**. That is what today is about.

</div>

---

# The fix in one slide · a per-parameter step size

<div class="insight">

**Big idea (2 sentences).** An *adaptive* optimizer gives every weight its own step size — small in steep directions, large in flat ones. **Adam** does this by keeping **two running averages per weight** and dividing one by the square root of the other.

</div>

Two averages, kept **per weight** (EMA = exponential moving average = a "recent average"):

- $m_t$ — **mean gradient**. This is just **momentum** from L4 · smooths *which way* to step.
- $v_t$ — **mean squared gradient**. Measures *how big / how noisy* gradients have been · the per-parameter scale.

The update — before any fine print:
$$\theta_t = \theta_{t-1} - \eta\,\frac{m_t}{\sqrt{v_t}+\epsilon}$$

Read it aloud · *smoothed gradient, divided by how big gradients have been.* Big/noisy direction → **damped**; small/quiet direction → **amplified**. Everything else today just **earns, fixes, or schedules these two averages.**

---

# One picture · big gradient → small step

![w:840px](figures/lec05/svg/per_coordinate_stepsize.svg)

Dividing by $\sqrt{v}$ **inverts** the gradient sizes · the steep direction (big $g$) gets a small step, the flat direction (small $g$) gets a big step — automatically, one scale per coordinate.

---

# Four questions

1. How do we get a **per-parameter** learning rate?
2. What is **Adam** actually doing, piece by piece?
3. Why is L2 "broken" inside Adam, and how does **AdamW** fix it?
4. What **schedule** should you use, and why do Transformers need warmup?

---

# Try these · interactives for L05

▶ [optimizer-race](https://nipunbatra.github.io/interactive-articles/optimizer-race/) (SGD vs momentum vs Adam trajectories on 2D loss) · [optimizers-beyond](https://nipunbatra.github.io/interactive-articles/optimizers-beyond/) (AdaGrad / RMSProp / AdamW side by side) · [lr-schedule-visualizer](https://nipunbatra.github.io/interactive-articles/lr-schedule-visualizer/) (cosine + warmup).

---

# Pop quiz · which optimizer would you bet on?

You are training a 1B-parameter Transformer on text · gradients are sparse for embeddings, dense for attention, and large for a few outlier weights.

<div class="popquiz">

(a) SGD with momentum 0.9.
(b) AdaGrad.
(c) AdamW with warmup + cosine.
(d) Plain Adam, constant LR.

Stop and decide. By the end of today the answer should be obvious — and you'll know **why** each of the others fails.

</div>

The exact same loss-curve shape would be diagnosed differently for each optimizer. That diagnostic is what this lecture trains.

---

# Bridge from ES 335 · you already know this

| From ES 335 | What changes today |
|---|---|
| you hand-tuned one $\eta$ for the whole model | every parameter earns its *own* effective $\eta$ |
| L2 penalty $=$ weight decay (identical in SGD) | inside Adam they silently diverge — AdamW repairs it |
| one fixed LR for the whole run | a *schedule*: warmup, then cosine decay |
| momentum (L4) = EMA of gradients | Adam adds a second EMA — of *squared* gradients |

<div class="keypoint">

You tuned one learning rate by hand. What if every parameter chose its own?

</div>

---

<!-- _class: section-divider -->

### PART 1

# The family tree

You've seen *what* Adam outputs ($m_t$, $v_t$, and $\eta\, m_t/\sqrt{v_t}$). Now let's **earn each piece** · SGD → AdaGrad → RMSProp → Adam.

---

# The lineage

![w:920px](figures/lec05/svg/optimizer_family_tree.svg)

---

# How do we give each parameter its own LR?

<div class="insight">

**Analogy · audio mixing board.** Imagine you're an engineer with a giant board of thousands of knobs (parameters).

- Some knobs are very sensitive — you've already moved them a lot. Make tiny adjustments now.
- Other knobs you've barely touched. You can afford to turn them aggressively.

</div>

**AdaGrad's idea** · keep a *total movement history* for each knob. The more a knob has moved, the smaller its future turns. This is the core idea behind every adaptive optimizer — RMSProp, Adam, AdamW all refine it.

---

# AdaGrad · build the update step-by-step

For a *single* parameter $\theta_i$:

1. **Gradient** at step $t$ for parameter $i$: $g_{t,i}$.
2. **Running sum of squared gradients** (the "history"):
$$G_{t,i} = G_{t-1,i} + g_{t,i}^2,\qquad G_{0,i} = 0$$
3. **Standard SGD** would do $\theta_{t,i} = \theta_{t-1,i} - \eta\,g_{t,i}$.
4. **AdaGrad** divides $\eta$ by $\sqrt{G_{t,i} + \epsilon}$:
$$\theta_{t,i} = \theta_{t-1,i} - \frac{\eta}{\sqrt{G_{t,i} + \epsilon}}\,g_{t,i}$$
$\epsilon = 10^{-8}$ avoids division-by-zero. Vector form (element-wise):
$$G_t = G_{t-1} + g_t^2, \qquad \theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{G_t + \epsilon}}\, g_t$$

Each parameter gets its **own** effective LR.

---

# Worked numeric · AdaGrad on two parameters

$\theta_1$ (dense, small gradients) and $\theta_2$ (sparse, occasionally huge). $\eta=0.1$.

<div class="columns">
<div>

### $\theta_1$ · steady $g = 0.2$

- $t=1$: $G = 0.04 \Rightarrow \eta_\text{eff} = 0.1/\sqrt{0.04} = 0.5$
- $t=2$: $G = 0.08 \Rightarrow \eta_\text{eff} \approx 0.354$
- $t=10$: $G = 0.40 \Rightarrow \eta_\text{eff} \approx 0.158$

LR shrinks gently as updates accumulate.

</div>
<div>

### $\theta_2$ · sparse: $g_1 \approx 0$, $g_2 = 5.0$

- $t=1$: $G \approx 0 \Rightarrow$ tiny update (nothing to update toward)
- $t=2$: $G = 25 \Rightarrow \eta_\text{eff} = 0.1/5 = 0.02$

One big gradient → LR for $\theta_2$ collapses immediately. Past sparsity protected; future moves are careful.

</div>
</div>

---

# AdaGrad's problem · LR decays to zero

![w:920px](figures/lec05/svg/adagrad_decay.svg)

---

# How do we fix AdaGrad's dying LR?

<div class="insight">

**Analogy · perfect vs. fading memory.** AdaGrad has *infinite memory* — every gradient since step 1 is in $G_t$. After a million steps, $G_t$ is huge and the effective LR is essentially zero.

What if we gave it a **fading memory**, like momentum? An **EMA of squared gradients** weights recent gradients more than old ones. That's **RMSProp**.

</div>

---

# RMSProp · AdaGrad with a fading memory (2012)

Replace AdaGrad's accumulator with an EMA $v_t$:

| | AdaGrad | RMSProp |
|---|---|---|
| Update | $G_t = G_{t-1} + g_t^2$ (forever) | $v_t = \beta_2\,v_{t-1} + (1-\beta_2)\,g_t^2$ |
| Behaviour | Grows without bound | Stabilizes |

Update rule:
$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t} + \epsilon}\,g_t$$
Typical $\beta_2 = 0.999$ — keep 99.9% of old, mix in 0.1% of new.

---

# Worked numeric · AdaGrad vs. RMSProp

Constant gradient $g = 2.0$, $\eta = 0.1$, $\beta_2 = 0.9$ for clarity.

<div class="columns">
<div>

### AdaGrad · keeps shrinking

- $t=1$: $G=4 \Rightarrow \eta_\text{eff} = 0.05$
- $t=2$: $G=8 \Rightarrow \eta_\text{eff} \approx 0.035$
- $t=10$: $G=40 \Rightarrow \eta_\text{eff} \approx 0.016$
- $t=100$: $\eta_\text{eff} \to 0$

</div>
<div>

### RMSProp · stabilizes

- $t=1$: $v=0.4 \Rightarrow \eta_\text{eff} \approx 0.158$
- $t=2$: $v=0.76 \Rightarrow \eta_\text{eff} \approx 0.115$
- $t \to \infty$: $v \to g^2 = 4 \Rightarrow \eta_\text{eff} = 0.05$

</div>
</div>

RMSProp's effective LR **converges to a positive value** rather than decaying to zero.

---

# The big idea · Adam = Momentum + RMSProp

Combine the best of both worlds:

1. **From momentum** — EMA of gradients gives a smoother, less noisy direction. Call it $m_t$ (first moment).
2. **From RMSProp** — EMA of *squared* gradients gives a per-parameter scale. Call it $v_t$ (second moment).

**Adam does both at once.**

---

# Adam · Momentum + RMSProp

![w:920px](figures/lec05/svg/adam_components.svg)

---

# Trajectories · SGD vs momentum vs Adam

![w:920px](figures/lec05/svg/optimizer_trajectories.svg)

---

# Adam · the full update

Same $m_t$, $v_t$, and update you saw up front — now with the **bias-correction fine print** ($\hat{m}_t,\hat{v}_t$) added (Part 2 explains why).

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

| step | $g_t$ | $m_t$ | $\hat m_t$ | $v_t$ | $\hat v_t$ | update |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | $2.0$ | $0.200$ | $2.00$ | $0.0040$ | $4.0$ | $-10^{-3}\,\tfrac{2.00}{\sqrt{4.0}} = -0.0010$ |
| 2 | $1.0$ | $0.280$ | $1.47$ | $0.0050$ | $2.5$ | $-10^{-3}\,\tfrac{1.47}{\sqrt{2.5}} \approx -0.00093$ |
| 3 | $1.5$ | $0.402$ | $1.48$ | $0.0072$ | $2.4$ | $-10^{-3}\,\tfrac{1.48}{\sqrt{2.4}} \approx -0.00095$ |

Row 1 in full · $m_1 = 0.9(0) + 0.1(2.0) = 0.20$, $\hat m_1 = 0.20/(1-0.9^1) = 2.00$, $v_1 = 0.001(2.0^2) = 0.004$, $\hat v_1 = 0.004/(1-0.999^1) = 4.0$.

</div>

Note · the per-parameter update size is **roughly constant ($\sim 10^{-3}$)** even though the gradients change. That's Adam's per-parameter adaptive scaling at work.

---

<!-- _class: section-divider -->

### PART 2

# The bias-correction detail

Why we divide by $(1 - \beta^t)$

---

# The cold-start problem

Our moving averages $m_t$ and $v_t$ are **initialized at zero**.

<div class="insight">

**Analogy · making hot chocolate.** You start with a cup of hot water ($m_0 = 0$) and add one spoonful of cocoa (gradient $g_1$). The first sip is mostly water with a hint of cocoa — *biased toward the starting point*.

After many spoonfuls the mix is right. Bias correction is the math trick to "undilute" the early steps and get the right strength immediately.

</div>

---

# What goes wrong at $t = 1$?

Initialize $m_0 = 0$. At step 1:
$$m_1 = \beta_1 \cdot 0 + (1 - \beta_1)\, g_1 = 0.1\, g_1$$

The EMA is **10× smaller** than the true gradient — purely because it started from zero.

---

# The correction in one picture

![w:920px](figures/lec05/svg/adam_bias_correction.svg)

---

# ⭐⭐⭐ Optional · deriving the bias-correction factor

**Gist ·** unroll the EMA from zero → $m_t = g\,(1-\beta^t)$, so dividing by $(1-\beta^t)$ recovers $g$.

Unroll from $m_0 = 0$ with a roughly constant gradient $g_k = g$:

- $m_1 = (1-\beta_1)\,g$
- $m_2 = \beta_1(1-\beta_1)\,g + (1-\beta_1)\,g$
- $m_3 = \beta_1^2(1-\beta_1)\,g + \beta_1(1-\beta_1)\,g + (1-\beta_1)\,g$

The pattern is a **geometric series** with $t$ terms:
$$m_t = (1-\beta_1)\,g\sum_{i=0}^{t-1}\beta_1^i = (1-\beta_1)\,g\cdot\frac{1 - \beta_1^t}{1 - \beta_1} = g\,(1 - \beta_1^t)$$

Divide to recover $g$ ·  $\boxed{\hat{m}_t = \dfrac{m_t}{1 - \beta_1^t}}$  — same for $\hat{v}_t = v_t/(1-\beta_2^t)$.

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

Without correction $m_1 = (1-\beta_1)g = 0.1g$ (10× too small) — **but** $v_1 = (1-\beta_2)g^2 = 0.001g^2$ makes $\sqrt{v_1}$ ~32× too small too. They **partly cancel**: the raw first step $m_1/\sqrt{v_1} \approx 3.2\,\mathrm{sign}(g)$ is about **3× too large**, not small. Correcting *both* moments by $(1-\beta^t)$ gives the clean unit step $\hat m_1/\sqrt{\hat v_1} = \mathrm{sign}(g)$.

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

You set `weight_decay` — so why is L2 silently broken inside Adam?

---

# L2 · quick recap from ES 335

L2 regularization adds $\frac{\lambda}{2}\|\theta\|^2$ to the loss. Gradient contribution: $\lambda\, \theta$.

In plain SGD, this equals *weight decay* — each step shrinks weights by a factor of $(1 - \eta\lambda)$:

$$\theta_t = \theta_{t-1} - \eta\,\nabla \mathcal{L} - \eta \lambda\, \theta_{t-1}$$

For SGD these are the same. **Not so for Adam.**

---

# AdamW · the fix

![w:920px](figures/lec05/svg/adamw_vs_adam.svg)

<div class="insight">

**Big idea (2 sentences).** In Adam, the L2 penalty rides *through* the $1/\sqrt{\hat v}$ scaling, so heavily-updated weights barely get decayed. **AdamW** applies decay **directly to the weights**, uniformly — so it regularizes the way you intended.

</div>

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

# Four common schedules

![w:920px](figures/lec05/svg/lr_schedules.svg)

<div class="realworld">

▶ Interactive: sliders for peak LR and warmup — [lr-schedule-visualizer](https://nipunbatra.github.io/interactive-articles/lr-schedule-visualizer/).

</div>

---

# A schedule is coupled to your budget

<div class="keypoint">

Cosine and warmup+cosine need `total_steps` **up front** — the curve is stretched to fit the budget. Double the budget and the *same* schedule becomes a different experiment.

</div>

- Budget unknown / exploratory → constant, or constant-with-decay-on-plateau.
- Fixed budget, one shot → warmup + cosine (decay to ~0 by the end, or a small floor like peak/10 if you set one).
- Comparing two runs? Match their schedules *relative to budget*, not in absolute steps.

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

# ⭐⭐⭐ Optional · why are early gradients so chaotic?

**Gist ·** at step 1 a random net makes huge, wrong-way gradients *and* Adam's $\sqrt{\hat v_t}$ estimate is still garbage — large ÷ tiny = explosion. Warmup starts $\eta$ near 0 to survive it.

A randomly-initialized network knows **nothing** — and Transformers make it worse: attention can accidentally focus everything on one irrelevant token, producing enormous gradients on that head.

Two things go wrong **simultaneously** at step 1:

1. **Chaotic, large gradients** — the random net makes overconfident-but-wrong predictions (softmax assigns 99% to the wrong class) → massive corrective gradients.
2. **Adam's $\hat v_t$ is itself unreliable** — only a few batches seen; if they happen to be small, $\sqrt{\hat v_t}$ is tiny.

$$\text{Update} \propto \frac{\text{large } g_t}{\text{tiny } \sqrt{\hat{v}_t}} \;\Longrightarrow\; \text{EXPLOSION}$$

One explosive first step throws the weights into unrecoverable territory.

---

# Warmup · the safety valve

Linearly ramp $\eta$ from 0 over the first 1–10% of training:

- At step 1, $\eta \approx 0$ → tiny update no matter how crazy the gradient.
- This gives $m_t, v_t$ time to **stabilize** over many batches.
- By the time $\eta$ reaches its target, $\hat{v}_t$ is a reliable estimate.

After warmup, use cosine decay.

---

<!-- _class: section-divider -->

### PART 5

# What to actually use

Which optimizer, which LR, which schedule — for *your* model?

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

`lr = 3e-4` is not magic — it's the number to use when you don't want to think. For a real run, do the **LR finder** (Lecture 3). And the first row is no accident: on vision-from-scratch, well-tuned SGD-momentum still often edges out AdamW in final accuracy — an empirical pattern, not a theorem.

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

# Putting it all together · the L05 master sentence

<div class="math-box">

**Adam = momentum on $g$, RMSProp on $g^2$, with a bias-correction at the start.** AdamW peels weight-decay out of the adaptive scaling so it actually regularizes. A **schedule** wraps everything · warmup at the start (gradients are chaotic), cosine at the end (anneal toward a minimum).

</div>

| Symbol | Role | Update at step $t$ |
|:-:|:-:|:-:|
| $m_t$ | momentum | $\beta_1 m_{t-1} + (1-\beta_1) g_t$ |
| $v_t$ | per-param scale | $\beta_2 v_{t-1} + (1-\beta_2) g_t^2$ |
| $\hat m_t, \hat v_t$ | bias-corrected | $m_t / (1-\beta_1^t)$, $v_t / (1-\beta_2^t)$ |
| $\theta_{t+1}$ | step | $\theta_t - \eta\,\hat m_t / (\sqrt{\hat v_t} + \epsilon) - \eta\,\lambda\,\theta_t$ (AdamW) |

The final $-\eta\lambda\theta_t$ term — applied **directly to weights, not through $\hat v$** — is the only difference between Adam+L2 and AdamW. That single change is why AdamW is the 2026 default for every Transformer in the world.

---

# Pop quiz · revisit

The 1B-parameter Transformer? The answer is **(c) AdamW with warmup + cosine**.

<div class="keypoint">

(a) SGD · single LR can't cope with sparse vs dense gradients · slow.
(b) AdaGrad · LR collapses to zero on dense layers within 10k steps.
(c) ✓ AdamW + warmup absorbs the chaotic early gradients · cosine anneals.
(d) Plain Adam · L2 silently broken; no warmup → first-step explosions.

</div>

---

# Practice problems

<div class="math-box">

**P1.** Show that AdaGrad's effective LR for parameter $\theta_i$ at step $t$ is $\eta / \sqrt{\sum_{s=1}^{t} g_{s,i}^2}$. Argue why this monotonically decreases.

**P2.** A parameter has constant gradient $g = 1$. Compute Adam's $\hat m_t$ and $\hat v_t$ at $t = 1, 2, 10, 100$. Verify the bias-correction recovers $\hat m_1 = g$ and $\hat v_1 = g^2$.

**P3.** State the **exact** difference between Adam-with-L2 and AdamW. Show that for a fixed gradient and fixed $\hat v_t$ they give different updates.

**P4.** A Transformer is trained for $T = 100\text{k}$ steps. Sketch the cosine schedule with $T_\text{warm} = 2000$ warmup steps and peak LR $= 6\!\times\!10^{-4}$. Give the LR at $t = 0, 2000, 50000, 100000$.

**P5.** Why does Adam with $\beta_2 = 0.999$ need bias correction at $t=1$ but not at $t=10000$? Compute $1 - \beta_2^t$ for both.

**P6.** You ran AdamW with `weight_decay=0` for 50 epochs. Train loss is 0.001, val loss is 0.7. Name two changes from this lecture (not L06's regularization) that would help and why.

</div>

---

# What breaks · symptom → suspect → test

| Symptom | Suspect | Fastest test |
|---|---|---|
| Adam diverges in the first ~100 steps of fine-tuning | no warmup — $\hat v_t$ still unreliable | add a 500-step linear warmup, rerun |
| `weight_decay` set, yet huge weights & overfitting | plain Adam: L2 term gets divided by $\sqrt{\hat v}$ | switch the class to **AdamW**, same $\lambda$ |
| Loss plateaus early, never reaches baseline | constant LR — no annealing phase | add cosine decay; final LR should be ~peak/10 |
| Embeddings learn, dense layers stall (or vice versa) | one global LR on mixed sparse/dense gradients | log per-layer grad norms; swap SGD → AdamW |
| AdamW "regularizes" nothing | `weight_decay` silently left at its 0 default | print `opt.param_groups[0]['weight_decay']` |

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

---

# The one-sentence takeaway

<div class="insight">

**Adam gives every parameter its own learning rate.**

Momentum smooths *where* to step ($m_t$); the second moment decides *how far* ($\sqrt{\hat v_t}$); warmup buys time for both estimates to become trustworthy.

</div>

*Next lecture: the optimizer is tuned — now we stop the model from memorizing. Regularization, including the kinds with no penalty term at all.*
