---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Training Deep Networks in Practice

## Lecture 3 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Write a **PyTorch training loop** by hand.
2. Apply the **seven-rung debug ladder** · data → overfit → scale up.
3. Use **LR finder** to pick a learning rate in 100 steps.
4. Set up **mixed-precision** training with autocast.
5. Diagnose **training curves** · underfit, sweet spot, overfit.
6. Save and load **checkpoints** correctly (state_dict, not pickle).

---

# Recap · what we have so far

- **Depth works** with ResNets + He init + ReLU (L2).
- **PyTorch recipe** — forward, loss, zero_grad, backward, step (L1).

<div class="paper">

Today maps to **UDL Ch 6** (early — fitting models) and **Ch 8** (measuring performance).

</div>

Theory is done. Today is entirely **practical**.

Four questions:
1. How does PyTorch actually wire this together?
2. How do you build a data pipeline that doesn't bottleneck the GPU?
3. What is the **debugging procedure** for when training fails?
4. How do you do **error analysis** (Ng-style) after training?

---

<!-- _class: section-divider -->

### PART 1

# The PyTorch stack

What happens when you type `model(x)`

---

# `nn.Module` · the container

![w:900px](figures/lec03/svg/nn_module_structure.svg)

---

# Parameter registration · a common footgun

<div class="warning">

If you store a tensor as `self.foo = torch.tensor(...)`, PyTorch will **not** track it as a parameter. No gradients, won't get moved to GPU, won't be saved.

- Learnable: use `nn.Parameter(torch.tensor(...))`.
- Non-learnable but device-bound: use `self.register_buffer('foo', ...)` (running stats, position codes).

</div>

```python
self.temperature = nn.Parameter(torch.ones(1))   # ← learnable
self.register_buffer('running_mean',
                     torch.zeros(dim))            # ← moves with .to() but no gradient
```

---

# How does PyTorch know the derivatives?

It watches you compute.

<div class="insight">

**Analogy · Baking a Cake and Asking "Why is it so sweet?"**
A friend watches you bake. They write down every step: *"1 cup flour, 2 cups sugar, 1 egg…"*. When you taste the cake and say *"too sweet,"* your friend looks at the notes (the **tape**) and says: *"The sugar had the biggest impact — use less and the sweetness drops the most."*

</div>

Autograd is that friend. It records every operation in a **computation graph**. When you call `.backward()`, it walks back through the graph to see how much each parameter contributed to the loss.

---

# Autograd · let's build a graph by hand

Trace a simple computation: $\;L = a \cdot x + b$ with $a=2,\ x=3,\ b=1$. Parameters: $a, b$. Input: $x$.

<div class="columns">
<div>

### Forward pass

- $c = a \cdot x = 2 \cdot 3 = 6$
- $L = c + b = 6 + 1 = 7$

PyTorch builds the graph **as we compute**.

</div>
<div>

### Backward pass (chain rule)

- $\partial L / \partial L = 1$
- $L = c + b \Rightarrow \partial L/\partial c = 1,\ \partial L/\partial b = 1$ ✓
- $c = a \cdot x \Rightarrow \partial L/\partial a = \partial L/\partial c \cdot x = 1 \cdot 3 = 3$ ✓

</div>
</div>

`loss.backward()` does this for every parameter, no matter how deep the graph.

---

# Autograd · the dynamic tape

![w:920px](figures/lec03/svg/autograd_tape.svg)

---

# Worked numeric · a 2-layer backward step

Compute $\partial L / \partial w_1$ for $L = (\mathrm{relu}(w_2 h) - y)^2$ where $h = \mathrm{relu}(w_1 x)$.
Let $x = 2,\ y = 1,\ w_1 = w_2 = 0.5$.

<div class="columns">
<div>

### Forward

- $a_1 = w_1 x = 0.5 \cdot 2 = 1.0$
- $h = \mathrm{relu}(a_1) = 1.0$
- $a_2 = w_2 h = 0.5$
- $\hat y = \mathrm{relu}(a_2) = 0.5$
- $L = (\hat y - y)^2 = 0.25$

</div>
<div>

### Backward (chain rule)

$\dfrac{\partial L}{\partial w_1} = \dfrac{\partial L}{\partial \hat y}\cdot\dfrac{\partial \hat y}{\partial a_2}\cdot\dfrac{\partial a_2}{\partial h}\cdot\dfrac{\partial h}{\partial a_1}\cdot\dfrac{\partial a_1}{\partial w_1}$

- $\partial L/\partial \hat y = 2(\hat y - y) = -1.0$
- ReLU' = 1 (both gates open)
- $\partial a_2/\partial h = w_2 = 0.5$
- $\partial a_1/\partial w_1 = x = 2$

</div>
</div>

$\partial L / \partial w_1 = (-1.0)(1)(0.5)(1)(2) = \boxed{-1.0}$

---

# Two safety habits

<div class="columns">
<div>

### `torch.no_grad()` for evaluation

```python
model.eval()
with torch.no_grad():
    for x, y in val_loader:
        pred = model(x)
        ...
```

Skips tape construction · faster · less memory.

</div>
<div>

### `.detach()` to stop gradient flow

```python
target = model_old(x).detach()
loss = mse(model_new(x), target)
```

Treats the tensor as a constant in the graph.

</div>
</div>

---

<!-- _class: section-divider -->

### PART 2

# The data pipeline

CPU loads while the GPU computes

---

# Dataset → DataLoader → device

![w:920px](figures/lec03/svg/dataloader_pipeline.svg)

---

# Writing a custom `Dataset`

```python
from torch.utils.data import Dataset

class TinyImageDataset(Dataset):
    def __init__(self, paths, labels, tfm=None):
        self.paths, self.labels, self.tfm = paths, labels, tfm

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = PIL.Image.open(self.paths[i]).convert('RGB')
        if self.tfm: img = self.tfm(img)
        return img, self.labels[i]
```

Two methods. That is the entire contract.

---

# DataLoader flags that matter

```python
loader = DataLoader(dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True)
```

<div class="insight">

Rule of thumb · `num_workers ≈ 4 × num_GPUs`. `pin_memory=True` when using CUDA. `persistent_workers=True` when epochs are short.

</div>

---

# Why not just use full precision (FP32)?

Modern GPUs are highways for numbers. To go faster: build a bigger highway (new GPU) **or make the cars smaller**.

- **FP32** (single precision) — the standard car. 32 bits per number.
- **FP16** (half precision) — a motorcycle. 16 bits.
  - **Pro:** 2× as many fit on the highway → ≈2× faster training, half memory.
  - **Con:** small range. Big numbers don't fit; gradients **overflow** to infinity → run dies.

We want the speed of 16 bits **without** losing FP32's range.

---

# Precision vs. range · the trade-off

<div class="insight">

**Analogy · measuring with rulers**

- **FP32** → 1-metre stick with millimetre marks. Big range, fine precision. Default.
- **FP16** → 15 cm ruler with millimetre marks. **Very precise**, but **range is tiny**. Try to measure a person's height — ruler runs out instantly.
- **BF16** → 1-metre stick with **centimetre** marks. **Same range as FP32**, less precision. Sacrifices a little precision for huge range.

</div>

For deep learning, **range matters far more than ultra-fine precision**. BF16 almost never overflows.

---

# Mixed precision · BF16 is the 2026 default

![w:920px](figures/lec03/svg/mixed_precision.svg)

---

# Mixed precision · why BF16 > FP16

<div class="columns">
<div>

### FP16

- Range · ±65,504
- Precision · 3-4 decimal digits
- Easy to **overflow** during loss computation
- Needs **loss scaling** to keep gradients in range

</div>
<div>

### BF16

- Range · ±10³⁸ (same as FP32)
- Precision · 2-3 decimal digits
- Never overflows
- **No loss scaling needed**

</div>
</div>

<div class="keypoint">

BF16 · trades precision for range. Same memory as FP16. Available on NVIDIA A100+, AMD MI200+, TPU v3+. Default on any modern LLM training.

</div>

```python
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)   # most ops in BF16
    loss = criterion(output, target)
loss.backward()            # grads in BF16 too
```

---

# What if the batch doesn't fit in your GPU?

Your GPU handles batches of 64, but the model trains better with batch 256.

<div class="insight">

**Analogy · polling a small town.** You want the average opinion of 256 people, but only 64 fit in the room.
1. Bring in 64. Tally — but **do not declare a result**.
2. Bring in the next 64. Add to the tally.
3. Repeat for groups 3 and 4.
4. After all 256 → compute the average → make a decision.

</div>

**Gradient accumulation** does exactly this with batches: it processes several **micro-batches**, sums their gradients, and updates the weights **once** at the end.

---

# Simulating a big batch · gradient accumulation

The key identity: **gradient of a sum = sum of gradients.**
$$\nabla\!\left(\sum_i L_i\right) = \sum_i \nabla L_i$$

Effective batch 256, GPU fits 64 → $K = 4$ accumulation steps.

```python
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / K   # divide so we average over K
    loss.backward()                     # ADDS to existing grads in .grad
    if (i + 1) % K == 0:
        opt.step()                      # update once per K micro-batches
        opt.zero_grad()                 # then clear for next big batch
```

Two crucial details: divide loss by $K$ (so we get the **mean**, matching a real batch of 256), and only call `step()` / `zero_grad()` after $K$ micro-batches.

---

# Worked numeric · gradient accumulation

Update weight $w$ for $L = (wx - y)^2$. LR $\alpha = 0.1$. Effective batch 2, micro-batch 1, $K=2$, start $w=0$.

<div class="columns">
<div>

### Micro-batch 1 · $x=2,\ y=1$

- $L_1 = (0\cdot 2 - 1)^2 / 2 = 0.5$
- $\nabla L_1 = \dfrac{x}{K} \cdot 2(wx-y) = \dfrac{2}{2}\cdot 2(-1) = -2$
- `loss.backward()` → `w.grad = -2`
- *No `step()`, no `zero_grad()`.*

</div>
<div>

### Micro-batch 2 · $x=3,\ y=2$

- $L_2 = (0\cdot 3 - 2)^2 / 2 = 2.0$
- $\nabla L_2 = \dfrac{3}{2}\cdot 2(-2) = -6$
- `loss.backward()` → `w.grad = -2 + (-6) = -8`

</div>
</div>

**Update step.** $w_{\text{new}} = 0 - 0.1 \cdot (-8) = 0.8$. Then `zero_grad()` resets `w.grad = 0`.

This is the **exact** update we'd get from one big batch of $\{(2,1),(3,2)\}$.

---

# Why we need gradient clipping

Some batches — especially in RNNs and Transformers — produce **ridiculously large** gradients. This is an **exploding gradient**.

<div class="insight">

**Analogy · learning to drive.** Your instructor gives small corrections: *"turn 5°,"* *"forward 10 cm."* Then suddenly screams **"TURN 10,000° LEFT!"**. Following that literally → smashed wall. Weights destroyed, restart training.

</div>

**Gradient clipping** is a safety rule:

> *"No matter what gradient is computed, never let the step be larger than `max_norm`."*

It prevents one bad batch from wrecking the entire run.

---

# How gradient clipping works

A gradient is a vector — direction × magnitude.

1. After `loss.backward()`, gather full gradient vector $g$.
2. Compute its **L2-norm** $\|g\|$.
3. Set threshold `max_norm` (e.g. 1.0).
4. If $\|g\| > \text{max\_norm}$, **rescale**:
   $$g_{\text{clipped}} = g \cdot \dfrac{\text{max\_norm}}{\|g\|}$$
5. Otherwise, leave $g$ alone.

**Direction is preserved — only step size is capped.**

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
```

---

# Worked numeric · gradient clipping

Two weights $w_1, w_2$. After `.backward()`: $g = [3.0,\ 4.0]$. Set `max_norm = 1.0`.

1. **Compute the norm.**
   $\|g\| = \sqrt{3^2 + 4^2} = \sqrt{25} = 5.0$
2. **Compare.** $5.0 > 1.0$ → must clip.
3. **Scaling factor.** $s = \text{max\_norm} / \|g\| = 1.0 / 5.0 = 0.2$
4. **Rescale.** $g_{\text{clipped}} = [3.0 \cdot 0.2,\ 4.0 \cdot 0.2] = [0.6,\ 0.8]$
5. **Verify.** $\|g_{\text{clipped}}\| = \sqrt{0.36 + 0.64} = 1.0$ ✓

The optimizer now uses $[0.6, 0.8]$ — same direction as $[3, 4]$ but a much safer step.

---

<!-- _class: section-divider -->

### PART 3

# The full training recipe

From zero to a trained model

---

# Training curves · the diagnostic language

![w:920px](figures/lec03/svg/training_curves_annotated.svg)

---

# The recipe · one function

```python
def train(model, loader_tr, loader_val, opt, loss_fn, n_epochs, device):
    best_val = float('inf')
    for ep in range(n_epochs):
        model.train()
        for x, y in loader_tr:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()

        val = evaluate(model, loader_val, loss_fn, device)
        print(f'epoch {ep:3d}  val_loss={val:.4f}')
        if val < best_val:
            best_val = val
            torch.save(model.state_dict(), 'best.pt')
```

Every real script is a variation on this. Keep it boring.

---

# Save / load correctly

<div class="columns">
<div>

### Save

```python
torch.save({
    'model':  model.state_dict(),
    'optim':  opt.state_dict(),
    'epoch':  ep,
    'config': cfg,
}, 'ckpt.pt')
```

</div>
<div>

### Load

```python
ckpt = torch.load('ckpt.pt',
                  map_location=device,
                  weights_only=True)
model.load_state_dict(ckpt['model'])
opt.load_state_dict(ckpt['optim'])
```

</div>
</div>

<div class="warning">

Don't `torch.save(model)` — it pickles the class, which breaks across refactors. Always save `state_dict`.

</div>

---

<!-- _class: section-divider -->

### PART 4

# The debugging ladder

(Karpathy's recipe — don't skip a rung)

---

# The ladder

![w:880px](figures/lec03/svg/debug_ladder.svg)

---

# Rung 3 · overfit one batch

![w:900px](figures/lec03/svg/overfit_one_batch.svg)

---

# Debug checklist · why won't loss go down? · part 1

If you can't even overfit a single batch, something is fundamentally broken.

- **LR too large** → loss explodes to `NaN`. **Fix:** divide LR by 10.
- **LR too small** → loss barely moves. **Fix:** multiply LR by 10.
- **Double softmax** → model ends with `nn.Softmax` AND you use `nn.CrossEntropyLoss`. The loss already includes softmax → applied twice → gradients muffled. **Fix:** remove `nn.Softmax` from the model; output raw **logits**.
- **Forgot `opt.zero_grad()`** → gradients from previous batches pile up; updates point in nonsense directions. **Fix:** add `opt.zero_grad()` at the start of each step.

---

# Debug checklist · why won't loss go down? · part 2

- **Dead ReLUs** → input to a ReLU is always negative → output 0, gradient 0; the unit can never learn. **Fix:** LeakyReLU, lower LR, or He init.
- **Frozen parameters** → a layer has `requires_grad=False`. It will never update. **Fix:** assert `param.requires_grad` for each layer you intend to train.
- **Wrong label shape / dtype** → `CrossEntropyLoss` wants class **indices** (e.g. `[3, 0, 1]`, dtype `torch.long`). One-hot floats break it. **Fix:** check `.shape` and `.dtype` of the label tensor.
- **Data not normalized** → raw pixels in $[0, 255]$ → unstable training. **Fix:** apply `ToTensor` + `Normalize`.

<div class="realworld">

Karpathy: *"Become one with the data."* Before touching the model, print shapes, dtypes, ranges, label balance, and a few random examples.

</div>

---

# How do we find a good starting LR?

The learning rate is **the most important** hyperparameter.

- Too high → training diverges (loss explodes).
- Too low → training crawls (or gets stuck).

We want the **highest LR that is still stable** — without dozens of trial-and-error runs.

<div class="insight">

**Analogy · pushing a cart up a hill.** Find the hardest you can push without tipping. Start with a tiny push and gradually increase. More force → more speed. At some point the cart wobbles. The best push is **just before** the wobble.

</div>

The LR finder does this with your learning rate.

---

# The LR finder algorithm

A short fake training session (≈100 steps) sweeps over LRs.

1. Start at LR = $10^{-7}$.
2. Each mini-batch → forward, backward, step.
3. After each step, **multiply LR by ~1.05**.
4. Record loss at each step.
5. Plot loss vs. LR (log-x).

**How to read the plot:**

- Find the region where loss **drops fastest** (the steepest downward slope).
- Loss eventually shoots back up — that's where training is unstable.
- **Pick an LR one order of magnitude before the minimum**, in the steep-descent zone.

---

# Rung 5 · the learning-rate finder

![w:920px](figures/lec03/svg/lr_finder.svg)

---

# Worked numeric · reading the LR plot

Suppose the LR finder gives:

| LR | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ | $1.0$ |
|----|-----------|-----------|-----------|-----------|-------|
| Loss | 3.2 | 2.5 | **1.1** (steepest drop) | 0.9 (minimum) | 2.8 (diverging) |

**Analysis.** Minimum loss is at LR $= 0.1$ — **don't pick this**, it's already on the edge of diverging. The steepest descent is at LR $\approx 10^{-2}$. Rule of thumb: pick **one order of magnitude before the minimum**.

**Good starting LR · $10^{-2}$.** Use this as `max_lr` for one-cycle training.

---

# Full recipe · the 7 rungs

1. **Inspect the data.** Shapes, dtypes, label balance, a few random samples.
2. **Dumb baseline.** Predict-the-mean / most-frequent-class.
3. **Overfit one batch.** Loss → ≈ 0, or fix the bug.
4. **Small subset.** 1–5% of data; full pipeline sanity.
5. **LR finder.** Pick a sensible learning rate.
6. **Full run + regularize.** Weight decay, augmentation, dropout.
7. **Error analysis.** Which examples fail, and why?

**Never skip a rung.** If rung 3 fails, debug at rung 3 — don't go tune hyperparams at rung 6.

---

<!-- _class: section-divider -->

### PART 5

# Error analysis (Ng style)

After training — what next?

---

# You have a model. Val accuracy is 82%.

<div class="popquiz">

**Q.** Which of these is most useful?

(a) Try a bigger model.
(b) Train for more epochs.
(c) Sample 100 val mistakes and categorize them.
(d) Tune the learning rate.

</div>

---

# Answer · (c)

<div class="insight">

**Ng's rule.** Before adding complexity, *look at the errors*. Nearly always you will find a dominant failure category — fixing it moves val accuracy far more than architectural churn.

</div>

---

# Error analysis · categorize, then prioritize

![w:900px](figures/lec03/svg/error_analysis_buckets.svg)

---

# Ceiling analysis — for pipelines

![w:920px](figures/lec03/svg/ceiling_analysis.svg)

---

<!-- _class: section-divider -->

### PART 6

# Reproducibility

The small things that save you weeks later

---

# Five layers of reproducibility

![w:900px](figures/lec03/svg/reproducibility_stack.svg)

---

# Seeds and determinism · code

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Only if bit-exact reproduction matters (slower):
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
```

Seed set at the start; every experiment records its seed in the config.

---

<!-- _class: summary-slide -->

# Lecture 3 — summary

- **`nn.Module`** auto-registers parameters; use `nn.Parameter` / `register_buffer` explicitly.
- **Autograd** is a dynamic tape built every forward; wrap eval in `torch.no_grad()`.
- **DataLoader tuning** — `num_workers`, `pin_memory`, `persistent_workers`.
- **Mixed precision (BF16)** — 2× speed, half memory. Default on Ampere+.
- **Debug ladder** — never skip a rung. Overfit one batch first.
- **Error analysis (Ng)** — categorize failures *before* scaling up.
- **Reproducibility** — code + config + data + seed + env. Weeks saved.

### Read before Lecture 4

**Prince** — Ch 6 (fitting models), Ch 8 (measuring performance). Free at [udlbook.github.io](https://udlbook.github.io/udlbook/).

### Next lecture

Optimization properly — loss landscapes, momentum, Nesterov.

<div class="notebook">

**Notebook 3a** · `03a-training-recipe.ipynb` — full training loop with checkpointing, LR finder.
**Notebook 3b** · `03b-debug-ladder.ipynb` — all seven rungs applied to a small task.

</div>
