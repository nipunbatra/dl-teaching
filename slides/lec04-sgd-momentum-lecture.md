---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# SGD, Momentum, Nesterov

## Lecture 4 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Identify **ravines** and **saddles** in high-dimensional loss.
2. Show why **vanilla SGD** oscillates across ravines.
3. Derive **momentum** as EMA of gradients.
4. Explain **Nesterov's lookahead** and its convergence rate payoff.
5. Pick **β** appropriately (0.9 default; when to adjust).
6. Diagnose an optimizer failure from training curves.

---

# Recap · where we are

- **Deep networks** are trainable with ResNets + He init + ReLU.
- **PyTorch recipe** — forward, loss, zero_grad, backward, step.
- **Debugging ladder** — overfit one batch, LR finder, error analysis.

<div class="paper">

Today maps to **UDL Ch 6** · fitting models (SGD, momentum, acceleration).

</div>

One piece we glossed over: **the optimizer**. Today we open that box.

---

# Four questions for today

1. What does the loss landscape *look like*, really?
2. Why does vanilla SGD oscillate on it?
3. How does momentum fix the oscillation?
4. What does Nesterov's *lookahead* add on top?

---

<!-- _class: section-divider -->

### PART 1

# The loss landscape

What makes neural-net optimization hard

---

# Gradient descent — picture

![w:900px](figures/lec04/svg/gd_intuition.svg)

---

# Three kinds of critical points

![w:920px](figures/lec04/svg/saddle_minima.svg)

---

# The ravine problem — κ elongates the basin

![w:920px](figures/lec04/svg/condition_number.svg)

---

# Why vanilla SGD oscillates on ravines

![w:900px](figures/lec04/svg/loss_landscape_ravine.svg)

---

# Ravine · zig-zag vs glide

![w:920px](figures/lec04/svg/ravine_zigzag.svg)

---

# The condition number · in numbers

Consider the quadratic $\mathcal{L}(\theta) = \frac{1}{2}(10 \theta_1^2 + \theta_2^2)$. Hessian eigenvalues $\lambda_1 = 10$, $\lambda_2 = 1$. **Condition number** $\kappa = 10$.

<div class="math-box">

For GD to converge, $\eta < 2/\lambda_\max = 0.2$.
Rate of contraction along $\theta_1$: $|1 - \eta \lambda_1| \le 1 - 0.2 \cdot 10 \cdot ? $...
Along $\theta_2$: $|1 - \eta| \approx 0.8$ per step.

</div>

The small direction moves **10× slower** than the large one wants to. Vanilla GD must pick a step small enough for $\lambda_\max$; every other direction then crawls.

**In deep nets $\kappa$ is often 10³–10⁶.** This is why momentum, adaptive LR, and normalization all help — they rescale so that $\kappa$ matters less.

---

# Why high-dim loss is mostly saddles

Random-matrix intuition: for a Hessian in $D$ dimensions with i.i.d. eigenvalue signs, probability all positive is $2^{-D}$.

<div class="keypoint">

At $D = 10^8$ parameters, **true local minima are exponentially rare**. Almost every critical point is a saddle — some directions go down, some up.

</div>

Good news · you almost never get stuck at a true local minimum.
Bad news · you *do* get stuck near saddles, where the gradient is small in many directions. This is where momentum's memory saves you — it carries you past the flat region in the direction you were already going.

---

# Mini-batch noise is not always bad

Gradient from a batch is a *noisy estimate* of the full gradient.

- **Bad:** adds variance to each step.
- **Good:** helps escape saddle points and shallow local minima.
- **Good (more):** implicit regularization from noise is part of why SGD generalizes.

<div class="insight">

Larger batch → less noise → worse generalization in practice. "Linear-scaling" rule: if you 2× the batch, 2× the learning rate.

</div>

---

<!-- _class: section-divider -->

### PART 2

# Momentum

The single most important change to SGD

---

# Momentum · the heavy-ball analogy

<div class="keypoint">

Vanilla SGD is a **short-sighted hiker** · only looks at the slope under their feet. In a narrow canyon they zig-zag wildly.

Momentum turns the hiker into a **heavy ball rolling down the hill**. The ball's inertia smooths out the zig-zags and carries it through small bumps and flat spots.

</div>

Algorithmically · keep an exponentially-weighted average of past gradients · use that as the update direction. The next slide turns this analogy into one line of math.

---

# Momentum = EMA of gradients

![w:920px](figures/lec04/svg/momentum_ema.svg)

---

# The physical intuition

Replace **position updates** with **velocity updates**. A ball rolling down the valley:

- accumulates speed in consistent directions
- averages out back-and-forth from noise

Formally — keep an exponential moving average of past gradients.

---

# Momentum · numerical trace

Let gradients in a ravine look like: $g_t = [\pm 1, 0.1]$ (flipping sign on $\theta_1$ every step, small consistent push on $\theta_2$).

<div class="math-box">

With $\beta = 0.9$, EMA settles to:
- $v_{t,1}$ · average of $\pm 1$ oscillations → **near zero**
- $v_{t,2}$ · average of $0.1$ → **0.1** (preserved)

</div>

Vanilla SGD zig-zags $\pm 1$ on $\theta_1$. Momentum cancels that out — direction 2 gets all the step budget. **Zig-zag in, drift out.**

The same EMA mechanism shows up in Adam (L5), batch-norm running stats, and target networks in RL. One primitive, many uses.

---

# Momentum = one more hyperparameter?

Yes — but a forgiving one.

| β | effective memory | behavior |
|:-:|:-:|:-:|
| 0.5 | 2 steps | barely smooths |
| 0.9 | 10 steps | **sensible default** |
| 0.95 | 20 steps | slower to respond to curvature changes |
| 0.99 | 100 steps | heavy; needs gradient clipping |

<div class="insight">

Most practitioners set $\beta = 0.9$ once and never touch it again. The knob you tune is $\eta$.

</div>

---

# Momentum · the update

<div class="math-box">

**SGD with momentum (Polyak 1964)**

$$\mathbf{v}_t = \beta\, \mathbf{v}_{t-1} + (1 - \beta)\, \nabla \mathcal{L}(\theta_{t-1})$$

$$\theta_t = \theta_{t-1} - \eta\, \mathbf{v}_t$$

Typical $\beta = 0.9$ · EMA with effective memory $\frac{1}{1-\beta} = 10$ steps.

</div>

PyTorch's `SGD(..., momentum=0.9)` uses an equivalent form.

---

# What momentum fixes

![w:900px](figures/lec04/svg/sgd_zoo.svg)

<div class="realworld">

▶ Interactive: race SGD, momentum, Adam on a 2D quadratic — [optimizer-race](https://nipunbatra.github.io/interactive-articles/optimizer-race/).

</div>

---

# A concrete example · MNIST

Same model, same LR, after 10 epochs:

| Optimizer | Train loss | Val accuracy |
|-----------|-----------|--------------|
| SGD | 0.42 | 92.1% |
| SGD + momentum | 0.13 | 97.6% |

Momentum is a **free 5 points** on MNIST. The single highest-value change to SGD.

---

# Momentum in PyTorch · one line

```python
# vanilla SGD
opt = torch.optim.SGD(model.parameters(), lr=0.01)

# SGD + momentum  — the sensible default
opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

<div class="realworld">

Never use vanilla SGD without momentum for a deep network. It is a Lego brick, not a finished optimizer.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Nesterov accelerated gradient

Evaluate the gradient one step ahead

---

# Nesterov · driving with a longer-range view

<div class="keypoint">

Standard momentum is like driving by looking at the road right in front of your bumper · you steer based on what's directly under you.

**Nesterov** is like looking *down the road*. You first imagine where momentum is taking you, look at the slope **at that future point**, and steer based on that.

</div>

The result · less overshoot near valley walls · cleaner approach to the minimum · provably better convergence rate (in the convex case).

---

# Classical vs Nesterov

![w:920px](figures/lec04/svg/nesterov_lookahead.svg)

---

# Nesterov · the update

<div class="math-box">

**Nesterov accelerated gradient (NAG, 1983)**

Evaluate the gradient at the *lookahead* point $\theta_{t-1} - \eta \beta\, \mathbf{v}_{t-1}$:

$$\mathbf{v}_t = \beta\, \mathbf{v}_{t-1} + (1 - \beta)\, \nabla \mathcal{L}\!\left(\theta_{t-1} - \eta \beta\, \mathbf{v}_{t-1}\right)$$

$$\theta_t = \theta_{t-1} - \eta\, \mathbf{v}_t$$

</div>

One change: *where* the gradient is computed.

---

# Why it helps

If the lookahead overshoots, the gradient at that point **points back** — correction kicks in *before* you commit to the full step.

Less overshoot near curvature changes, slightly faster convergence.

<div class="math-box">

**Theoretical payoff** (convex, smooth case): optimal convergence rate $O(1/t^2)$ vs GD's $O(1/t)$.

</div>

In practice on deep nets, the gain is modest but free.

---

# A geometric way to see Nesterov

<div class="columns">
<div>

### Classical momentum

1. Take step $\eta \beta v_{t-1}$ (momentum part)
2. Add a correction based on gradient *at start*

If the gradient at the start was misleading, you've committed before knowing.

</div>
<div>

### Nesterov

1. Tentatively move $\eta \beta v_{t-1}$ to a *lookahead* point
2. Measure the gradient *there*
3. Use that gradient for the real step

Information from the landscape you're about to visit, not the one you're leaving.

</div>
</div>

That's it · the same update, just measured at a smarter location.

---

# Nesterov in PyTorch · one flag

```python
opt = torch.optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      nesterov=True)    # ← the flag
```

That is the entire cost of using it.

---

<!-- _class: section-divider -->

### PART 4

# Practical recommendations

What to actually use

---

# Current 2026 practice

| Use-case | Optimizer | Reason |
|----------|-----------|--------|
| CNN from scratch | **SGD + momentum + Nesterov** | best test accuracy on vision |
| Fine-tuning anything | AdamW (L5) | robust across LR |
| Transformer training | AdamW + warmup + cosine | field default |
| Debugging a new model | Adam first | faster to iterate |

<div class="insight">

Image researchers often prefer SGD-Momentum for final runs — it tends to find flatter minima that generalize slightly better. Everyone else uses AdamW.

</div>

---

# A common mistake

<div class="warning">

**Q.** Student sets `momentum=0.99` because "more is better." Loss diverges. Why?

Higher $\beta$ = longer memory. If curvature changes abruptly (early training), a heavy memory keeps you pushing in stale directions *after* the gradient has reversed → overshoot.

**Rule of thumb:** keep $\beta \in [0.9, 0.95]$. Go higher only with strong gradient clipping + warmup.

</div>

---

# Momentum changes the effective LR

Momentum changes the *effective* learning rate:

$$\eta_\text{eff} \approx \frac{\eta}{1 - \beta}$$

If you double $\beta$ from $0.9$ to $0.95$, effective LR doubles — you may need to halve $\eta$.

<div class="realworld">

Practical recipe: fix $\beta = 0.9$; use the LR finder to pick $\eta$. Revisit $\beta$ only if training is unstable.

</div>

---

# Debugging optimizer failures

Common symptoms and fixes:

| Symptom | Likely cause | Fix |
|:-:|:-:|:-:|
| Loss → NaN after step 1 | LR too high, fp16 overflow | halve $\eta$, enable gradient clipping |
| Loss oscillates (±) | ravine + momentum too low | raise $\beta$ to 0.9 |
| Loss plateaus for hundreds of steps | stuck near saddle | raise $\beta$ or switch to Adam |
| Loss drops then climbs | overfitting (not optimizer) | add weight decay, lower $\eta$ |
| Training slower than Keras example | no momentum | add `momentum=0.9` |

<div class="insight">

Most "my network doesn't train" bugs are optimizer-level. The debug ladder from L3 + this table catches ~90% of them in practice.

</div>

---

<!-- _class: summary-slide -->

# Lecture 4 — summary

- **Loss landscapes** — ravines, saddles, ill-conditioning. High-dim is mostly saddles.
- **Vanilla SGD** serves one direction at a time → oscillates across narrow valleys.
- **Momentum** = EMA of gradients; damps zig-zag, reinforces consistent directions.
- **Nesterov** evaluates the gradient at the lookahead point — a free small speed-up.
- **In practice** · SGD+momentum(+Nesterov) for vision, AdamW for everything else.

### Read before Lecture 5

**Prince** — Ch 6 §6.4–6.6. Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

**Adam, AdamW, and learning-rate schedules** — per-parameter adaptive LR, bias correction, decoupled weight decay, warmup + cosine.

<div class="notebook">

**Notebook 4** · `04-optimizer-race.ipynb` — implement SGD, momentum, Nesterov from scratch; animate trajectories on a 2D quadratic and Rosenbrock.

</div>
