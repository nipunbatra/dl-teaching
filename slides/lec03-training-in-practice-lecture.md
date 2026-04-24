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

# Autograd · the dynamic tape

![w:920px](figures/lec03/svg/autograd_tape.svg)

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

# Gradient accumulation + clipping

<div class="columns">
<div>

### Gradient accumulation

```python
for i, (x, y) in enumerate(loader):
    loss = criterion(model(x), y) / K
    loss.backward()
    if (i + 1) % K == 0:
        opt.step()
        opt.zero_grad()
```

Effective batch = `micro × K`.

</div>
<div>

### Gradient clipping

```python
loss.backward()
torch.nn.utils.clip_grad_norm_(
    model.parameters(), 1.0)
opt.step()
```

Insurance against spikes for Transformers / RNNs.

</div>
</div>

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

# Checklist when overfit-one-batch fails

- LR too small or too large
- `softmax` applied AND `CrossEntropyLoss` (double softmax)
- Forgot `optimizer.zero_grad()`
- Dead ReLUs — all post-activation values 0
- Frozen parameters — check `.requires_grad`
- Wrong label shape / dtype (int vs float)
- Data not normalized (inputs in $[0, 255]$ instead of scaled)

<div class="realworld">

Karpathy: *"Become one with the data."* Before looking at the model, print shapes, dtypes, ranges, label balance, a few random examples.

</div>

---

# Rung 5 · the learning-rate finder

![w:920px](figures/lec03/svg/lr_finder.svg)

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
