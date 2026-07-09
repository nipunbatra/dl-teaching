---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Optimization II: Momentum to Adam

## Lecture 6 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Plain SGD takes the same step in every direction. Real loss surfaces are **lopsided** — steep across the ravine, flat along it — so one global step size is always wrong somewhere.

<div class="keypoint">

**Every modern optimizer is SGD with two extra memories.**
A running average of the gradient (**momentum**) fixes *which way* to step; a running average of the squared gradient (**RMSProp**) fixes *how far*. **Adam** keeps both. That is the optimizer you will use for every model in this course.

</div>

In L5 you derived the step $\theta\leftarrow\theta-\eta\nabla J$ and saw it zig-zag on a ravine. Today we make that step **fast and nearly hands-free.**

$$\text{SGD} \rightarrow \underbrace{\text{momentum}}_{\text{EMA of } g} \rightarrow \underbrace{\text{AdaGrad}\to\text{RMSProp}}_{\text{EMA of } g^2} \rightarrow \underbrace{\text{Adam}}_{\text{both}} \rightarrow \underbrace{\text{AdamW + schedule}}_{\text{the default}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L7** — weight decay is AdamW's decoupled term, and L1's L2-prior
- **L8+** — every CNN, RNN, Transformer trains with AdamW
- **L14–L18** — LLM pretraining *is* AdamW + warmup + cosine, at scale
- **every lab** — `torch.optim.AdamW(lr=3e-4)` is the starting line

</div>
<div>

**Borrowed from earlier** (we build on, not repeat):
- the GD/SGD step, the ravine, saddles — **L5**
- L2 penalty $=$ Gaussian prior on weights — **L1**
- EMA / exponentially-weighted averages — **Maths for ML**

</div>
</div>

<div class="insight">

L5 was the *engine* (plain GD, then SGD). L6 is the *tuning* — momentum, adaptive rates, schedules — that turns a fiddly engine into the one-line default you'll stop thinking about.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Why plain SGD is slow and fiddly

---

# Recap from L5 · the ravine

A stretched bowl (condition number $\kappa=\lambda_{\max}/\lambda_{\min}\gg1$) is **steep across the valley, nearly flat along it.** A single $\eta$ small enough to be stable across the walls is far too small to make progress along the floor — so SGD bounces wall-to-wall, creeping forward:

![w:620px](figures/lec04/svg/ravine_zigzag.svg)

<div class="insight">

This is the single most common failure of vanilla SGD on real losses. Two things are broken: the **direction** wobbles sideways, and one **step size** can't serve both axes. Momentum fixes the first; adaptive rates fix the second.

</div>

---

# Two knobs vanilla SGD gets wrong

<div class="columns">
<div>

**Problem 1 — direction.**
Consecutive gradients point *across* the valley, alternating sign. Their raw sum zig-zags. We want to **average them** so the sideways bounces cancel and the consistent downhill component survives.
→ **momentum** (Part 2)

</div>
<div>

**Problem 2 — step size.**
The steep axis needs a *small* step; the flat axis needs a *large* one. One global $\eta$ can't do both. We want a step size that is **small where gradients are big, large where they're small** — per parameter.
→ **AdaGrad / RMSProp / Adam** (Part 3)

</div>
</div>

<div class="keypoint">

Both fixes are the same trick applied twice: **keep a running average.** Average the gradient → momentum. Average the squared gradient → a per-parameter scale. Adam is literally both averages at once.

</div>

---

<!-- _class: section-divider -->

## Part 2 · Momentum — an EMA of gradients

---

# Intuition: a heavy ball, not a hiker

Plain SGD is a hiker who re-decides direction from scratch at every step — so it lurches with every noisy gradient. Replace it with a **heavy ball** rolling downhill: it has *inertia*, so it keeps rolling along the consistent downhill direction and can't reverse instantly when a single gradient points sideways.

![w:760px](figures/lec04/svg/momentum_ema.svg)

The ball's velocity is an **exponentially-weighted average (EMA)** of the recent gradients — recent ones count most, old ones fade.

---

# Intuition · an EMA is just a smoother

The velocity $v_t=\beta v_{t-1}+(1-\beta)g_t$ is the **exponentially-weighted average** Ng uses to smooth a noisy temperature series: each new reading nudges the running average a little, so spikes get ironed out and the underlying trend shows through.

<div class="columns">
<div>

**How far back does it remember?**
The weights decay like $\beta,\beta^2,\beta^3,\dots$, so the average effectively spans the last
$$\frac{1}{1-\beta}\ \text{gradients.}$$
- $\beta=0.9 \Rightarrow \sim\!10$
- $\beta=0.98 \Rightarrow \sim\!50$
- $\beta=0.5 \Rightarrow \sim\!2$ (barely smooths)

</div>
<div>

**Higher $\beta$ = smoother but laggier.**
More memory irons out more noise — but the average also reacts *slower* to a genuine change of direction. The exact trade-off of any moving average: stability vs. responsiveness.

</div>
</div>

<div class="insight">

Momentum gets this for free: a jittery stream of gradients goes in, a smooth velocity comes out. Everything else today — RMSProp, Adam — is this **one smoothing tool**, applied to $g$ or to $g^2$.

</div>

---

# The momentum update rule

<div class="math-box">

**Momentum (EMA form, as in Ng's Course 2)**
$$v_t = \beta\, v_{t-1} + (1-\beta)\, g_t \qquad(\text{velocity} = \text{EMA of gradients})$$
$$\theta_t = \theta_{t-1} - \eta\, v_t \qquad(\text{step along the velocity, not the raw gradient})$$

</div>

- $\beta=0.9$ is the standard default — the velocity remembers roughly the **last $\tfrac{1}{1-\beta}=10$ gradients.**
- $\beta=0 \Rightarrow$ plain SGD (no memory). $\beta\to1 \Rightarrow$ a very heavy ball that ignores the newest gradient almost entirely.
- The classic "heavy-ball" form $v_t=\beta v_{t-1}+g_t$ is the same idea; it folds a factor $1/(1-\beta)$ into the effective step size (Appendix).

<div class="insight">

Same downhill step as L5 — but taken along a **smoothed** direction instead of the raw, jittery one.

</div>

---

# Intuition · momentum averages away mini-batch noise

In L5 each mini-batch gave only a **noisy** estimate of the true gradient — the descent direction jitters even on a smooth bowl. Averaging is the textbook cure for noise, and the EMA is exactly that average.

![w:640px](figures/lec04/svg/gradient_variance.svg)

<div class="keypoint">

Two failures, one mechanism. On a **ravine** the EMA cancels the sideways *oscillation*; on **noisy SGD** it cancels the random *variance*. Either way the consistent signal survives while the noise averages toward zero — so the step you actually take points more reliably downhill than any single gradient did.

</div>

---

# The one worked numeric: momentum damps the ravine

Two directions, $\beta=0.9$, velocities start at $0$. **Across** the valley you bounce off both walls, so gradients alternate $+2,-2,+2,\dots$; **along** the floor the gradient is steady at $+0.3$.

<div class="columns">
<div>

**Across (alternating $+2,-2$)**
- $v_1 = 0.1(+2) = 0.20$
- $v_2 = 0.9(0.20)+0.1(-2) = -0.02$
- $v_3 = 0.9(-0.02)+0.1(+2) = 0.18$

$|v|\approx 0.02\text{–}0.18$, vs. raw $|g|=2$ → **damped ~10–100×.**

</div>
<div>

**Along (steady $+0.3$)**
- $v_1 = 0.03$
- $v_2 = 0.9(0.03)+0.1(0.3) = 0.057$
- $v_3 = 0.9(0.057)+0.1(0.3) = 0.081$

$v$ **builds toward** the true gradient $0.3$ — consistent progress.

</div>
</div>

<div class="keypoint">

The sign-flipping wall gradients **cancel** in the average; the steady floor gradient **survives** and accumulates. Because the sideways oscillation is now tiny, you can safely **raise $\eta$** — which is where the real speed-up comes from.

</div>

---

# Nesterov: look before you leap

Standard momentum evaluates the gradient **where you are**, then adds the velocity. **Nesterov** first takes the momentum jump to a *look-ahead* point $\theta+\beta v$, and evaluates the gradient **there** — so it can correct a step that's about to overshoot *before* committing to it.

![w:720px](figures/lec04/svg/nesterov_lookahead.svg)

<div class="insight">

A ball that sees the wall coming and brakes early. In practice Nesterov gives a small, reliable edge and is the default `nesterov=True` on SGD-momentum for vision-from-scratch.

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** On a ravine you use momentum with $\beta=0.9$, $v_0=0$. Across the valley the first two gradients are $g_1=+4,\ g_2=-4$ (you bounced off both walls).

(a) Compute the velocities $v_1$ and $v_2$.
(b) Plain SGD would step by $-\eta g_2$. Compare $|v_2|$ to $|g_2|$ — by what factor is the across-valley motion damped?
(c) *One sentence:* why does damping this direction let you use a **larger** $\eta$?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$$v_1 = 0.9(0) + 0.1(+4) = 0.4$$
$$v_2 = 0.9(0.4) + 0.1(-4) = 0.36 - 0.4 = -0.04$$

(b) Momentum uses $|v_2| = 0.04$; SGD would use $|g_2| = 4$. That's a **100× smaller** across-valley step — the two opposite gradients almost perfectly cancel in the EMA.

<div class="keypoint">

(c) Vanilla SGD's ceiling on $\eta$ is set by the **steep** axis — go bigger and it diverges there. Momentum shrinks the steep-axis motion, so you can raise $\eta$ until the **flat** axis finally moves — faster overall, without exploding across the walls.

</div>

---

# See momentum in action

<div class="notebook">

**🎛 Interactive · optimizer race** — SGD vs momentum vs Adam trajectories on the same 2-D loss; watch momentum glide down the ravine while plain SGD zig-zags. *(Interactive Lab · `interactive-articles/optimizer-race`)*

</div>

<div class="notebook">

**🎛 Interactive · SGD on a ravine** — drag the condition number $\kappa$ and step size; see exactly when plain GD zig-zags and momentum smooths it out. *(Interactive Lab · to be authored at `~/git/interactive/articles/sgd-ravine`)*

</div>

---

<!-- _class: section-divider -->

## Part 3 · Per-parameter learning rates: AdaGrad → RMSProp → Adam

---

# One global learning rate is a compromise

Momentum fixed the *direction*, but still uses **one $\eta$ for every parameter.** Loss surfaces don't cooperate:

- **Steep direction** — gradients large → a big step *overshoots* and bounces.
- **Flat direction** — gradients tiny → a small step *crawls* and wastes time.

One global $\eta$ can't win both. Same story with **sparse vs dense** parameters: a rare-word embedding sees large, occasional gradients; a dense hidden weight sees small, constant ones — they want different step sizes.

<div class="keypoint">

The fix: give **every parameter its own step size** — automatically, from its own gradient history. That is what AdaGrad, RMSProp and Adam all do.

</div>

---

# The family tree

![w:900px](figures/lec05/svg/optimizer_family_tree.svg)

Each arrow adds exactly **one idea**: AdaGrad adds per-parameter scaling, RMSProp makes that scaling *forget*, Adam bolts momentum back on, AdamW fixes weight decay. We'll earn each one in turn.

---

# The shape of the fix: big gradient → small step

The recipe every adaptive optimizer shares: divide the step by (roughly) how big that parameter's gradients have been. Dividing by $\sqrt{v}$ **inverts** the gradient sizes.

![w:820px](figures/lec05/svg/per_coordinate_stepsize.svg)

The steep coordinate (big $g$) gets a **small** step; the flat coordinate (small $g$) gets a **large** step — one scale per coordinate, computed for free from the gradients you already have.

---

# Intuition · a private step size for every weight

One global $\eta$ is a single coach shouting the *same* stride length at every runner. Adaptive methods hire **one coach per weight**, each watching only *its own* gradient history.

<div class="columns">
<div>

**Steep / busy weight** — big, frequent gradients → coach says *"short steps, you keep overshooting."*

**Flat / quiet weight** — tiny gradients → coach says *"stride out, you're barely moving."*

</div>
<div>

**The sparse-feature win.**
A rare-word embedding fires only occasionally, so a global $\eta$ hardly ever moves it. Its private rate stays **large** between visits, so the one gradient it *does* see counts fully — this is the problem AdaGrad was invented for in sparse NLP.

</div>
</div>

<div class="insight">

Nothing here is hand-set: the per-weight rate is read straight off $\sqrt{v}$, the running size of that weight's own gradients. Adaptive methods **tune a million learning rates for you, every single step.**

</div>

---

# AdaGrad · accumulate the squared-gradient history

For a single parameter $\theta_i$, keep a running **sum of squared gradients** and divide $\eta$ by its square root:

<div class="math-box">

$$G_{t,i} = G_{t-1,i} + g_{t,i}^2,\qquad G_{0,i}=0$$
$$\theta_{t,i} = \theta_{t-1,i} - \frac{\eta}{\sqrt{G_{t,i}}+\epsilon}\, g_{t,i}$$

</div>

- $\epsilon=10^{-8}$ just avoids division by zero.
- A parameter that has seen **big/frequent** gradients builds a large $G$ → tiny future steps. One that's been **quiet** keeps a large effective step.
- The effective learning rate $\eta/\sqrt{G_{t,i}}$ is now **different for every $i$** — exactly what we wanted.

---

# Worked numeric · AdaGrad on two parameters

$\theta_1$ is dense with steady small gradients; $\theta_2$ is sparse with one big spike. $\eta=0.1$.

<div class="columns">
<div>

**$\theta_1$ · steady $g=0.2$**
- $t=1$: $G=0.04 \Rightarrow \eta_{\text{eff}}=0.1/0.2=0.5$
- $t=2$: $G=0.08 \Rightarrow \eta_{\text{eff}}\approx0.354$
- $t=10$: $G=0.40 \Rightarrow \eta_{\text{eff}}\approx0.158$

LR shrinks *gently* as history accumulates.

</div>
<div>

**$\theta_2$ · sparse, then $g=5.0$**
- $t=1$: $G\approx0 \Rightarrow$ (nothing to move on yet)
- $t=2$: $G=25 \Rightarrow \eta_{\text{eff}}=0.1/5=0.02$

One big gradient → its LR collapses at once. The rare parameter takes careful steps from then on.

</div>
</div>

Each parameter got its own effective rate from its own history — no hand-tuning.

---

# AdaGrad's flaw · the learning rate dies

$G_t$ **only ever grows** — it never forgets. After enough steps $\sqrt{G_t}$ is huge and the effective LR decays toward zero, so learning **stalls before it converges.**

![w:840px](figures/lec05/svg/adagrad_decay.svg)

Fine for a convex problem that finishes fast; fatal for a deep net trained for a million steps.

---

# RMSProp · give it a fading memory

Swap AdaGrad's *infinite* sum for an **EMA** of squared gradients — recent gradients count, ancient ones fade:

<div class="columns">
<div>

| | AdaGrad | RMSProp |
|---|---|---|
| accumulator | $G_t = G_{t-1}+g_t^2$ | $v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$ |
| memory | forever | last $\sim\!\tfrac{1}{1-\beta_2}$ |
| LR over time | $\to 0$ | **stabilizes** |

</div>
<div>

$$\theta_t = \theta_{t-1} - \frac{\eta}{\sqrt{v_t}+\epsilon}\, g_t$$

Typical $\beta_2=0.999$: keep 99.9% of the old estimate, mix in 0.1% of the new $g^2$.

</div>
</div>

Same per-parameter scaling as AdaGrad — but because $v_t$ tracks a *recent* window, it settles instead of blowing up, so the LR **never dies.**

---

# Worked numeric · AdaGrad vs RMSProp

Constant gradient $g=2.0$, $\eta=0.1$; use $\beta_2=0.9$ for clean arithmetic.

<div class="columns">
<div>

**AdaGrad · keeps shrinking**
- $t=1$: $G=4 \Rightarrow \eta_{\text{eff}}=0.05$
- $t=2$: $G=8 \Rightarrow \eta_{\text{eff}}\approx0.035$
- $t=10$: $G=40 \Rightarrow \eta_{\text{eff}}\approx0.016$
- $t\to\infty$: $\eta_{\text{eff}}\to 0$

</div>
<div>

**RMSProp · stabilizes**
- $t=1$: $v=0.4 \Rightarrow \eta_{\text{eff}}\approx0.158$
- $t=2$: $v=0.76 \Rightarrow \eta_{\text{eff}}\approx0.115$
- $t\to\infty$: $v\to g^2=4 \Rightarrow \eta_{\text{eff}}=0.05$

</div>
</div>

<div class="keypoint">

RMSProp's effective LR **converges to a positive value** ($\eta/|g|$); AdaGrad's decays to zero. That single change — remember *recently*, not forever — is the whole fix.

</div>

---

# Adam = momentum + RMSProp

Two averages you already understand, kept **per weight**, combined in one update:

![w:900px](figures/lec05/svg/adam_components.svg)

- $m_t$ — EMA of the gradient. This *is* **momentum** from Part 2: smooths *which way* to step.
- $v_t$ — EMA of the squared gradient. This *is* **RMSProp**: the per-parameter *how far*.

Read the update aloud: *smoothed gradient, divided by how big gradients have been.*

---

# Adam · the full update

Same $m_t$, $v_t$ — now with the **bias-correction** $\hat m_t,\hat v_t$ that repairs the cold start (explained on the next slide).

<div class="math-box">

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t \qquad\text{(1st moment — momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2 \qquad\text{(2nd moment — RMSProp)}$$
$$\hat m_t = \frac{m_t}{1-\beta_1^{\,t}}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}} \qquad\text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

**Defaults:** $\beta_1=0.9,\ \beta_2=0.999,\ \epsilon=10^{-8},\ \eta=10^{-3}$.

</div>

---

# Intuition · Adam in one mental picture

<div class="keypoint">

A **heavy ball** (momentum, $\hat m_t$) rolling downhill, fitted with a **shock absorber on every axis** (RMSProp, $\sqrt{\hat v_t}$) that stiffens wherever the ground has been bumpy.

</div>

Read the update $\theta_t=\theta_{t-1}-\eta\,\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$ as one sentence:

<div class="columns">
<div>

**Numerator $\hat m_t$ — *which way?***
The smoothed, noise-averaged direction. Inertia keeps it pointed along the consistent slope.

</div>
<div>

**Denominator $\sqrt{\hat v_t}$ — *how far?***
The recent gradient size on *this* axis. Big & bumpy → a short, cautious step; small & smooth → a long, confident one.

</div>
</div>

<div class="insight">

That is the whole optimizer. Everything left — bias correction, AdamW's decay, warmup — is fine print that protects this one idea at the very *start* and very *end* of training.

</div>

---

# Intuition · why divide by $\sqrt{v}$, not $v$?

$v_t$ tracks $g^2$, so it carries the **units of gradient-squared**. Taking the *square root* puts the denominator back in the **units of the gradient**, so the ratio $g/\sqrt{v}$ is (roughly) dimensionless and well-behaved.

<div class="columns">
<div>

**Divide by $v$ (too aggressive).**
A gradient of $10$ would be scaled by $1/100$; a gradient of $0.1$ by $1/0.01=100$. Step sizes swing over *four* orders of magnitude — wildly unstable.

</div>
<div>

**Divide by $\sqrt{v}$ (just right).**
The same gradients scale by $1/10$ and $1/0.1=10$ — a controlled $100\times$ spread. The update $g/\sqrt{v}$ lands near $\pm1$ for a *typical* gradient, so $\eta$ genuinely sets the step length.

</div>
</div>

<div class="insight">

This is why an Adam step stays $\approx\eta$ in magnitude no matter the raw gradient scale (you'll see it in Practice problem 2): the $\sqrt{\cdot}$ normalizes the gradient to roughly unit size *before* $\eta$ scales it.

</div>

---

# Trajectories · SGD vs momentum vs Adam

![w:900px](figures/lec05/svg/optimizer_trajectories.svg)

SGD zig-zags; momentum smooths the direction; Adam additionally rescales each axis, so it drives almost straight down the ravine floor. **Same start, same surface — the memory is what differs.**

---

# Why bias correction · the cold-start problem

$m_t$ and $v_t$ are **initialized at zero.** At step 1 the average is diluted toward that zero start:

$$m_1 = \beta_1\cdot 0 + (1-\beta_1)\,g_1 = 0.1\,g_1$$

— **10× too small**, purely because it started empty. Bias correction "un-dilutes" it.

![w:720px](figures/lec05/svg/adam_bias_correction.svg)

Dividing by $1-\beta_1^{\,t}$ exactly cancels this shrinkage (derivation in the Appendix), and fades to the identity once $t$ is large.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** One parameter, gradients $g_1=2.0,\ g_2=1.0$. Adam defaults ($\beta_1=0.9,\ \beta_2=0.999,\ \eta=10^{-3}$), $m_0=v_0=0$.

(a) Compute $m_1,\ v_1$ and the bias-corrected $\hat m_1,\ \hat v_1$.
(b) Give the first update $\Delta\theta_1 = -\eta\,\hat m_1/\sqrt{\hat v_1}$.
(c) Repeat for step 2 to get $\Delta\theta_2$. Is the step size roughly constant?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

<div class="math-box">

| $t$ | $g_t$ | $m_t$ | $\hat m_t$ | $v_t$ | $\hat v_t$ | $\Delta\theta_t=-\eta\,\hat m_t/\sqrt{\hat v_t}$ |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 1 | $2.0$ | $0.200$ | $2.00$ | $0.00400$ | $4.00$ | $-10^{-3}\,\tfrac{2.00}{\sqrt{4.0}}=-0.0010$ |
| 2 | $1.0$ | $0.280$ | $1.47$ | $0.00500$ | $2.50$ | $-10^{-3}\,\tfrac{1.47}{\sqrt{2.5}}\approx-0.00093$ |

Row 1 in full: $m_1=0.9(0)+0.1(2.0)=0.20$; $\hat m_1=0.20/(1-0.9)=2.00$; $v_1=0.001(2.0^2)=0.004$; $\hat v_1=0.004/(1-0.999)=4.0$.

</div>

<div class="keypoint">

The update is $\approx10^{-3}$ **both steps, even though the gradient halved** — Adam's $\sqrt{\hat v}$ normalizes the scale away, so $\eta$ sets the step size almost directly. That predictability is why $\eta=3\!\times\!10^{-4}$ "just works."

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** With $\beta_1=0.9$ and $m_0=0$, the first raw moment is $m_1=(1-\beta_1)g_1=0.1\,g_1$.

(a) By what factor is $m_1$ too small, and why?
(b) What is the correction factor $1/(1-\beta_1^{\,t})$ at $t=1$ vs $t=100$?
(c) So is bias correction an *always* fix or a *first-few-steps* fix?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

(a) $m_1=0.1\,g_1$ is **10× too small** — the EMA started from $0$, so one gradient can only fill in a $(1-\beta_1)=0.1$ fraction. Dividing by $1-\beta_1^{\,1}=0.1$ recovers $\hat m_1=g_1$.

(b) Correction factors:

$$t=1:\ \frac{1}{1-0.9}=10 \qquad t=100:\ \frac{1}{1-0.9^{100}}\approx 1.0000$$

<div class="insight">

(c) A **first-few-steps** fix. It's a factor of 10 at $t=1$, about $1.5$ at $t=10$, and indistinguishable from 1 by $t=100$ — it quietly switches itself off. Skip it and the *very first* Adam steps are the wrong magnitude, which is exactly when a fresh network is most fragile.

</div>

---

# See Adam in code

<div class="notebook">

**📓 Notebook · Adam from scratch** — implement $m_t,v_t$, bias correction and the update in ~15 lines; check your loop matches `torch.optim.Adam` step-for-step on a toy loss. *(ES 667 · `notebooks/06-optimization-momentum-adam.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · optimizers beyond SGD** — AdaGrad, RMSProp, Adam and AdamW side by side on the same surface; watch AdaGrad stall as its LR dies. *(Interactive Lab · `interactive-articles/optimizers-beyond` · zoo explainer to be authored at `~/git/interactive/articles/optimizer-zoo`)*

</div>

---

<!-- _class: section-divider -->

## Part 4 · AdamW — decoupled weight decay

---

# L2, one more time · a prior on the weights

From L1: adding $\tfrac{\lambda}{2}\lVert\theta\rVert^2$ to the loss is a **Gaussian prior** on the weights (MAP). Its gradient contribution is $\lambda\theta$, so in **plain SGD** the update literally shrinks the weights each step:

$$\theta_t = \theta_{t-1} - \eta\,\nabla\mathcal L - \eta\lambda\,\theta_{t-1}$$

The $-\eta\lambda\theta$ term is **weight decay** — for SGD, "add L2 to the loss" and "decay the weights" are *identical*. The trouble: **inside Adam they are not.**

---

# Why L2 breaks inside Adam — and how AdamW fixes it

![w:860px](figures/lec05/svg/adamw_vs_adam.svg)

<div class="insight">

If you fold L2 into the gradient, Adam divides it by $\sqrt{\hat v}$ along with everything else — so a weight with a **large gradient history barely gets decayed**, and the regularization you asked for is silently uneven. **AdamW** applies the decay **directly to the weights**, *outside* the adaptive scaling — uniform on every parameter, the way a prior should be.

</div>

---

# Adam vs AdamW · one-step worked numeric

Setup: $\theta=1.0,\ g=0.5,\ \lambda=0.1,\ \eta=10^{-3}$, and $\hat v=4.0$ from the second moment.

<div class="math-box">

**Adam (L2 folded into the gradient)**
- effective gradient $= g+\lambda\theta = 0.5+0.1 = 0.6$
- update $= -\eta\,(0.6)/\sqrt{4} = -0.00030$
- the decay term $\lambda\theta$ got **divided by $\sqrt{\hat v}$** — weakened on high-gradient weights

**AdamW (decoupled decay)**
- adaptive part $= -\eta\,(0.5)/\sqrt{4} = -0.00025$
- plus uniform decay $= -\eta\lambda\theta = -0.00010$
- total $= -0.00035$

</div>

In AdamW every parameter is decayed by the same fraction $\eta\lambda$, regardless of its gradient history — matching the L2-prior theory from L1. **Use AdamW.**

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** Two weights sit at $\theta=1.0$; you ask for weight decay $\lambda=0.1$, $\eta=10^{-3}$. Weight **A** has a large gradient history ($\sqrt{\hat v}=10$); weight **B** is quiet ($\sqrt{\hat v}=0.1$).

(a) Under **Adam with L2 folded into the gradient**, the decay part of the step is $-\eta\,(\lambda\theta)/\sqrt{\hat v}$. Compute it for A and B.
(b) Under **AdamW**, the decay is $-\eta\lambda\theta$, applied directly to the weight. Compute it for A and B.
(c) *One sentence:* which method regularizes the two weights *equally*, and why is that what the Gaussian (L2) prior from L1 actually asks for?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="math-box">

**(a) Adam (L2 in the gradient)** — decay $=-\eta\lambda\theta/\sqrt{\hat v}$:
$$\text{A:}\ -\frac{10^{-3}(0.1)(1)}{10}=-10^{-5} \qquad \text{B:}\ -\frac{10^{-3}(0.1)(1)}{0.1}=-10^{-3}$$
Weight A is decayed **100× less** than B — purely because it has bigger gradients.

**(b) AdamW (decoupled)** — decay $=-\eta\lambda\theta=-10^{-4}$ for **both** A and B.

</div>

<div class="keypoint">

(c) **AdamW** shrinks both weights by the same fraction $\eta\lambda$. A prior says *"all weights should be small"* — it knows nothing about gradient magnitudes, so the penalty **must** be uniform. Adam's $\sqrt{\hat v}$ leak makes the effective $\lambda$ silently depend on gradient history; AdamW removes the leak. That is the entire reason to prefer AdamW.

</div>

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

Typical weight decay: **LLM pretrain** `0.1` · **fine-tuning** `0.01–0.05` · **vision fine-tune** `0.001–0.01`. Leaving `weight_decay=0` gives you plain Adam with no regularization — a surprisingly common bug.

</div>

---

<!-- _class: section-divider -->

## Part 5 · Schedules, warmup & clipping

---

# One learning rate isn't enough over a whole run

Even with a perfect optimizer, the *best* $\eta$ changes over training: large early (cover ground), small late (settle into a minimum). A **schedule** varies $\eta$ across steps.

![w:900px](figures/lec05/svg/lr_schedules.svg)

**Step decay** drops at milestones; **cosine** anneals smoothly to near zero — the clean modern default.

---

# A schedule is coupled to your budget

<div class="keypoint">

Cosine (and warmup+cosine) need `total_steps` **up front** — the curve is stretched to fit the budget. Double the budget and the *same* schedule is a different experiment.

</div>

- **Budget unknown / exploratory** → constant, or decay-on-plateau.
- **Fixed budget, one shot** → warmup + cosine, decaying to $\sim$peak/10 (or to 0).
- **Comparing two runs?** Match schedules *relative to budget*, not in absolute steps.

---

# Why Transformers need warmup

At step 1 a random network makes wild, overconfident-wrong predictions → **huge gradients** — and Adam's $\sqrt{\hat v_t}$ estimate is still garbage from too few batches. Large $\div$ tiny $=$ an explosive first step that can wreck the run.

![w:900px](figures/lec05/svg/warmup_necessity.svg)

**Warmup** linearly ramps $\eta$ from $\approx0$ over the first 1–10% of steps: tiny updates while $m_t,v_t$ stabilize, then full speed once the estimates are trustworthy.

---

# Warmup + cosine · the Transformer default, in 5 lines

```python
import math
def lr_lambda(step, total_steps, warmup):
    if step < warmup:
        return step / warmup                       # linear warmup 0 → 1
    progress = (step - warmup) / (total_steps - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))  # cosine decay 1 → 0

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
```

<div class="realworld">

Warmup $=$ 1–10% of total steps (thousands for a big run — *not* 10). This 5-line schedule trains essentially every Transformer in production. **Interactive:** `interactive-articles/lr-schedule-visualizer` — drag peak LR and warmup length.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** You train for $T=10{,}000$ steps with peak LR $\eta_{\max}=3\times10^{-4}$, using **linear warmup over the first $W=1{,}000$ steps**, then **cosine decay to $0$**:
$$\eta(t)=\begin{cases}\eta_{\max}\,\dfrac{t}{W} & t<W\\[6pt] \eta_{\max}\cdot\tfrac12\Big(1+\cos\pi\,\dfrac{t-W}{T-W}\Big) & t\ge W\end{cases}$$

(a) $\eta$ at $t=500$ (inside warmup)?
(b) $\eta$ at $t=1000$ (warmup just ended)?
(c) $\eta$ at the cosine midpoint $t=5500$?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

<div class="math-box">

**(a) $t=500<W$:** $\ \eta=3\times10^{-4}\cdot\dfrac{500}{1000}=1.5\times10^{-4}$ — half of peak, still ramping.

**(b) $t=1000$:** warmup ends exactly at the *peak*; $\cos(0)=1\Rightarrow \eta=3\times10^{-4}$.

**(c) $t=5500$:** progress $=\dfrac{5500-1000}{10000-1000}=\dfrac{4500}{9000}=0.5$, so $\cos(\pi\cdot0.5)=0$ and
$$\eta=3\times10^{-4}\cdot\tfrac12(1+0)=1.5\times10^{-4}.$$

</div>

<div class="keypoint">

The curve **ramps up** to the peak over warmup, then **glides down** a half-cosine to $\approx0$ at $t=T$. Halfway through the decay you are back at half-peak — the same $\eta$ as halfway through warmup, but now heading *down*. Change $T$ and every one of these numbers moves: **the schedule is glued to the step budget.**

</div>

---

# See schedules & the optimizer race

<div class="notebook">

**🎛 Interactive · LR schedule visualizer** — drag peak LR, warmup length and total steps; watch warmup+cosine (and step decay) redraw, and read $\eta$ off at any step — the exact curve you computed in Practice problem 5. *(Interactive Lab · `interactive-articles/lr-schedule-visualizer`)*

</div>

<div class="notebook">

**🎛 Interactive · the optimizer race** — SGD, momentum, RMSProp, Adam and AdamW released from the *same* point on the *same* loss surface; scrub time and watch who reaches the minimum first and who stalls. *(Interactive Lab · `interactive-articles/optimizer-race`, `optimizers-beyond`)*

</div>

<div class="notebook">

**💡 Worth building — "watch the optimizers race on a loss surface"** — a flyover of a ravine / Rosenbrock surface with all five optimizers as colored balls, a live velocity vector and per-axis step-size read-out, and sliders for $\eta,\beta_1,\beta_2$. Turns every worked numeric in this lecture into something you can *see* move.

</div>

---

# Gradient clipping · cheap insurance

One freak batch can produce a giant gradient that blows up the weights. **Clip** the global gradient norm to a ceiling (say $1.0$) before the optimizer steps — direction preserved, magnitude capped.

![w:860px](figures/lec05/svg/gradient_clipping.svg)

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # before opt.step()
```

Standard for RNNs and Transformers — near-zero cost, saves runs from a single bad step.

---

# Defaults that actually work

| Model / regime | Optimizer | LR | Schedule |
|---|---|---|---|
| CNN from scratch | **SGD + momentum + Nesterov** | 0.1 | step decay |
| CNN fine-tune | AdamW | 1e-4 | cosine |
| Transformer pretrain | **AdamW** ($\beta_2=0.95$) | 3e-4 | **warmup + cosine** |
| LoRA fine-tune of an LLM | AdamW | 1e-4 – 3e-4 | cosine |
| Debugging a new idea | AdamW | 3e-4 | constant |

<div class="realworld">

`lr=3e-4` is the number to reach for when you don't want to think; tune it later with an LR finder. Note row 1: well-tuned SGD-momentum still often edges out AdamW on vision-from-scratch — an empirical pattern, not a theorem.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Adam is SGD with two memories — an EMA of the gradient (momentum) for *which way*, and an EMA of the squared gradient (RMSProp) for *how far* — with a bias correction at the start; AdamW decouples weight decay, and a warmup+cosine schedule wraps the whole run.**

</div>

- **Momentum**: $v_t=\beta v_{t-1}+(1-\beta)g_t$ — averages away the ravine's zig-zag.
- **Per-parameter scale**: AdaGrad (dies) → RMSProp (fading memory) → **Adam** = both averages, bias-corrected.
- **AdamW**: decay the weights directly, not through $\sqrt{\hat v}$ — the L1 L2-prior, done right.
- **Schedule**: warmup (survive chaotic early gradients) + cosine (anneal to a minimum); clip grads at 1.0.

**Next (L7):** regularization — weight decay as a prior, plus the tricks with no penalty term at all (dropout, early stopping, augmentation).

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and the primary references:

- **ES 667 optimization materials** — Adam / AdamW / schedule build-up, worked numerics & figure library (`figures/lec04`, `figures/lec05`), N. Batra.
- **Momentum practice problem & GD tutorial** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`optimization/tutorials/optimization.tex`).
- **Pedagogical framing** (exponentially-weighted averages as a smoother, momentum as EMA, per-parameter learning rates, batch→mini→stochastic, warmup) — A. Ng, *Deep Learning Specialization*, Course 2 (Improving Deep Neural Networks), Week 2.
- **Interactive explainers** (optimizer race, optimizers-beyond, LR-schedule visualizer) — *ES 667 Interactive Lab*, N. Batra.
- **Adam** — D. Kingma & J. Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015.
- **AdamW** — I. Loshchilov & F. Hutter, *Decoupled Weight Decay Regularization*, ICLR 2019.

All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · deriving the bias-correction factor

Unroll the EMA $m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t$ from $m_0=0$ with a roughly constant gradient $g_k=g$:

$$m_t = (1-\beta_1)\,g\sum_{i=0}^{t-1}\beta_1^{\,i} = (1-\beta_1)\,g\cdot\frac{1-\beta_1^{\,t}}{1-\beta_1} = g\,\big(1-\beta_1^{\,t}\big)$$

So the raw EMA is a factor $(1-\beta_1^{\,t})$ **below** the true gradient. Dividing it out recovers $g$ exactly:

$$\boxed{\hat m_t = \frac{m_t}{1-\beta_1^{\,t}}}\qquad\text{and identically}\qquad \hat v_t = \frac{v_t}{1-\beta_2^{\,t}}$$

At $t=1$ the factor is $1-\beta_1=0.1$ (the 10× we saw); as $t\to\infty$, $\beta_1^{\,t}\to0$ and the correction becomes the identity. *(Kingma & Ba 2015, §3.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · momentum's effective learning rate

The "heavy-ball" form $v_t=\beta v_{t-1}+g_t$ (no $1-\beta$ factor) exposes momentum's real speed-up. For a **constant** gradient $g$, unroll from $v_0=0$:

$$v_t = g\sum_{i=0}^{t-1}\beta^{\,i} \;\xrightarrow{\;t\to\infty\;}\; \frac{g}{1-\beta}$$

The velocity saturates at $g/(1-\beta)$, so the steady-state step is

$$\theta_t-\theta_{t-1} = -\eta\,v_t \;\longrightarrow\; -\frac{\eta}{1-\beta}\,g \;=\; -\eta_{\text{eff}}\,g,\qquad \eta_{\text{eff}}=\frac{\eta}{1-\beta}$$

With $\beta=0.9$ the effective learning rate is **$10\times$** the raw $\eta$ along any consistent-gradient direction — that is the acceleration down the ravine floor. In the EMA convention ($v_t=\beta v_{t-1}+(1-\beta)g_t$) this $1/(1-\beta)$ is folded in, which is why raising $\eta$ after adding momentum recovers the same behavior.
