---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Training Deep Nets in Practice

## Lecture 8 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Training a deep net that *works* is almost never about a cleverer architecture. It's a **diagnostic loop**: read one signal, apply the one fix that signal points to.

<div class="keypoint">

**Read the signal → apply the matching fix.**
The single most useful signal is the **train–val gap**. Both errors high → *bias*, fix by adding capacity. A big gap → *variance*, fix by adding data or regularization. A flat loss → a *wiring bug*, caught by overfitting one batch. This is a checklist, not magic.

</div>

You spent L1–L7 building the machine — losses, backprop, SGD, regularization. Today is the **operator's manual**: what to look at, in what order, when a run misbehaves.

$$\underbrace{\text{gap}}_{\text{bias/variance}} \;\rightarrow\; \underbrace{\text{ladder}}_{\text{overfit 1 batch}} \;\rightarrow\; \underbrace{\text{LR finder}}_{\text{one knob}} \;\rightarrow\; \underbrace{\text{error analysis}}_{\text{what to fix next}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's diagnostics in:**
- **L5** — the LR finder feeds the schedule you'll build
- **L6** — every *variance* fix (weight decay, dropout, augmentation) is a regularizer
- **L7–L9** — error analysis on real image models
- **L14–L18** — the same ladder, at LLM scale

</div>
<div>

**Borrowed from your own courses** (we build on, not repeat):
- **bias–variance, cross-validation, confusion matrix** — *ML (ES 335)*
- the framing of this entire lecture — **A. Ng, Course 3**
- the debug ladder — **A. Karpathy's recipe**

</div>
</div>

<div class="insight">

This is Andrew Ng's most practical course, distilled: **no new theory, just the discipline that separates a run that works from weeks of guessing.**

</div>

---

<!-- _class: section-divider -->

## Part 1 · Read the gap — bias vs variance

---

# The train–val gap is your compass

Two numbers, read together, tell you *what kind* of problem you have — before you change anything.

| Train error | Val error | Diagnosis | What it means |
|---|---|---|---|
| high | high | **high bias** | model too weak — *underfitting* |
| low | high (big gap) | **high variance** | memorizing — *overfitting* |
| low | low | **good fit** | ship it |
| high | low | *impossible* | a bug (leak / bad split) |

<div class="keypoint">

You cannot fix what you haven't diagnosed. The bias/variance read comes **first**, because the fix for one *makes the other worse*.

</div>

---

# The dartboard: four ways to be wrong

Think of every retrain (different data sample) as a dart. The bullseye is the truth $f(x)$.

![w:420px](figures/Lnew/svg/bias_variance_dartboard.svg)

<div class="insight">

**Bias** = are we *systematically* off? (off-center). **Variance** = are we *consistent* across training sets? (scatter). A deep net with too little data lands in the top-right: right on average, wildly different every seed.

</div>

---

# Where the two errors come from

For squared loss, the expected test error at $x$ decomposes into three non-negative pieces:

<div class="math-box">

$$\mathbb E\big[(y-\hat f(x))^2\big] = \underbrace{\big(\mathbb E[\hat f(x)]-f(x)\big)^2}_{\text{bias}^2} + \underbrace{\mathbb E\big[(\hat f(x)-\mathbb E[\hat f(x)])^2\big]}_{\text{variance}} + \underbrace{\sigma^2}_{\text{irreducible}}$$

</div>

- **Bias²** — how far the *average* model is from the truth. Shrinks with capacity.
- **Variance** — how much the model *wobbles* between training sets. Shrinks with data / regularization.
- **Irreducible** $\sigma^2$ — label noise. No model, however good, removes it.

That last term is why 100% is usually the wrong target. (Full derivation in the appendix.)

---

# How good is "good"? — human-level as a proxy for Bayes error

You can't reach below the **irreducible error** (Bayes error). Ng's trick: use **human-level performance** as a cheap stand-in for it, then split the gap.

<div class="math-box">

**avoidable bias** $=$ train error $-$ human level $\qquad$ **variance** $=$ val error $-$ train error

</div>

| Human | Train | Val | avoidable bias | variance | Work on |
|---|---|---|---|---|---|
| 1% | 8% | 10% | **7%** | 2% | **reduce bias** (bigger model, train longer) |
| 1% | 2% | 10% | 1% | **8%** | **reduce variance** (more data, regularize) |

<div class="keypoint">

Same 10% val error, *opposite* fixes. The bigger of the two gaps tells you where to spend the next week.

</div>

---

# Reading the learning curve

The gap is easiest to see as a picture: plot train and val loss vs. epochs and read off which regime you're in.

![w:930px](figures/lec03/svg/learning_curve_diagnosis.svg)

<div class="insight">

Both curves high and flat → **bias**. A widening gap (val turns up while train keeps falling) → **variance**. The shape *is* the diagnosis.

</div>

---

# One run sweeps through all three states

A single training run passes through underfit → sweet spot → overfit as epochs pass. The move: **stop at best-val** — just before the val curve turns back up.

![w:870px](figures/lec03/svg/training_curves_annotated.svg)

Early stopping is the cheapest variance fix there is: it costs nothing but a saved checkpoint.

---

# Read the signal → apply the matching fix

<div class="columns">
<div>

**High bias** (underfitting)
*train error itself is too high*
- bigger / deeper model
- train longer, better LR
- **less** regularization
- richer features / architecture

</div>
<div>

**High variance** (overfitting)
*train–val gap is too wide*
- more data / augmentation
- **more** regularization (L2, dropout)
- early stopping
- smaller model

</div>
</div>

<div class="warning">

Notice the mirror: *less* regularization fixes bias, *more* fixes variance. Push the wrong lever and you make things worse — which is why you diagnose first.

</div>

---

# See it, and play with it

<div class="notebook">

**📓 Notebook · bias–variance decomposition** — resample the training set many times, watch predictions scatter, and measure bias² and variance as model complexity grows. *(ML ES 335 · `notebooks/bias-variance.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · learning-curve diagnosis** — drag capacity, dataset size, and regularization; watch the train/val curves separate and re-read the diagnosis live. *(Interactive Lab · `interactive/articles/learning-curve-diagnosis`)*

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A classifier reports **train accuracy 99%, validation accuracy 80%**. (a) Diagnose: high bias or high variance? (b) List **three** fixes, and say which one costs the least.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**(a)** Train error 1%, val error 20% → a **19-point gap** with *low* train error. That is textbook **high variance (overfitting)** — the model fits the training set almost perfectly but fails to generalize. Bias is not the problem.

**(b)** Any three variance fixes:

<div class="keypoint">

1. **Early stopping** — free; just keep the best-val checkpoint. *(cheapest)*
2. **Regularization** — weight decay, dropout, or stronger data augmentation.
3. **More data** — collect more, or a smaller model if you can't.

</div>

Do **not** reach for a bigger model or more epochs — those reduce *bias*, and would widen this gap.

---

<!-- _class: section-divider -->

## Part 2 · The debugging ladder

---

# DL bugs are silent — so you need a procedure

In ES 335, a bug threw an exception. Here, the code runs fine and the **loss just doesn't go down**. Wrong label dtype, a double softmax, a frozen layer — all silent.

<div class="keypoint">

Because failure is silent, you don't *debug by staring* — you climb a **ladder** of ever-harder tests, and stop at the first rung that breaks. Never tune hyperparameters at rung 6 when rung 3 is failing.

</div>

Karpathy's first rule before any of it: **"become one with the data."** Print shapes, dtypes, value ranges, label balance, and eyeball a few raw examples. Most "model bugs" are data bugs.

---

# The ladder

![w:880px](figures/lec03/svg/debug_ladder.svg)

Each rung is a cheap, decisive test. If it fails, the bug lives *there* — fix it before climbing.

---

# Rung 3 · overfit ONE batch, first

Take **4 examples** and train until the loss is essentially **zero**. A model that can memorize 4 points has a working forward pass, loss, and gradient path.

![w:900px](figures/lec03/svg/overfit_one_batch.svg)

<div class="keypoint">

If you *cannot* drive the loss to ~0 on 4 examples, the bug is **structural** — wiring, not the learning rate. No optimizer setting will rescue a broken gradient path.

</div>

---

# Why won't the loss go down? · checklist (1/2)

Read each line as **symptom → what it means → fix**:

- **LR too large** → loss explodes to `NaN`. *Fix:* ÷10.
- **LR too small** → loss barely moves. *Fix:* ×10.
- **Double softmax** → `nn.Softmax` in the model *and* `nn.CrossEntropyLoss` (which already applies it) → gradients muffled. *Fix:* output raw **logits**, drop the softmax.
- **Forgot `zero_grad()`** → gradients accumulate across steps → nonsense direction. *Fix:* `opt.zero_grad()` each step.

---

# Why won't the loss go down? · checklist (2/2)

- **Dead ReLUs** → a unit's input is always negative → output 0, gradient 0, never recovers. *Fix:* LeakyReLU, lower LR, or He init.
- **Frozen parameters** → some layer has `requires_grad=False`. *Fix:* assert `requires_grad` on every layer you mean to train.
- **Wrong label shape/dtype** → `CrossEntropyLoss` wants class **indices** (`torch.long`), not one-hot floats. *Fix:* check `.shape` and `.dtype`.
- **Data not normalized** → raw pixels in $[0,255]$ → unstable. *Fix:* `ToTensor` + `Normalize`.

<div class="realworld">

Every item here is caught at **rung 3**. That's why overfitting one batch comes before everything else — it's the cheapest test that isolates *wiring* from *tuning*.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** Your loss is flat for 100 steps. Why does the recipe say **"overfit a single batch of 4"** *before* you touch the learning rate? What would a *successful* overfit rule out, and what would a *failed* one tell you?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

Overfitting one batch asks the most basic question: **can this machine reduce loss at all?** It's decisive and costs seconds (4 examples).

<div class="columns">
<div>

**If it succeeds** → forward pass, loss, and gradients are all wired correctly. The bug is now *definitely* in optimization or data scale — **now** the LR finder is the right tool.

</div>
<div>

**If it fails** → the fault is **structural** (double softmax, frozen layer, wrong loss, no `zero_grad`). Tuning the LR here is guessing — no learning rate fixes a broken gradient path.

</div>
</div>

<div class="keypoint">

Order matters: the LR is only meaningful *once you know the machine can learn.* Tuning it first is optimizing a circuit you haven't checked is connected.

</div>

---

<!-- _class: section-divider -->

## Part 3 · The learning-rate finder

---

# The learning rate is the one knob that matters most

Too high → training diverges (loss → `NaN`). Too low → it crawls, or stalls. We want the **highest LR that is still stable** — without dozens of trial runs.

<div class="insight">

**Analogy · pushing a cart uphill.** Start with a tiny push and increase steadily. More force → more speed — until the cart *wobbles*. The best push is **just before** the wobble. The LR finder does exactly this, automatically.

</div>

---

# The LR finder algorithm

One short fake training run (~100 steps) sweeps the LR from tiny to huge:

<div class="math-box">

1. Start at LR $=10^{-7}$, pick an end (e.g. $10$) and $n\approx100$ steps.
2. Each mini-batch → forward, backward, step.
3. After each step, **multiply the LR** by a constant $r=(\text{end}/\text{start})^{1/(n-1)}$ — a *geometric* sweep.
4. Record the (smoothed) loss; **stop on `NaN`**. Then **restore** the original weights.
5. Plot loss vs. LR on a **log-x** axis.

</div>

**How to read it:** pick an LR in the **steepest-descent** region — typically **3–10× below** where the loss bottoms out (the minimum itself is already on the edge of diverging).

---

# Reading the LR plot

![w:920px](figures/lec03/svg/lr_finder.svg)

The useful region is the steep downhill *before* the valley — not the bottom, which is already unstable.

---

# Worked numeric · reading the plot

Suppose the finder returns:

| LR | $10^{-4}$ | $10^{-3}$ | $10^{-2}$ | $10^{-1}$ | $1.0$ |
|---|---|---|---|---|---|
| Loss | 3.2 | 2.5 | **1.1** (steepest drop) | 0.9 (minimum) | 2.8 (diverging) |

<div class="keypoint">

The minimum is at LR $=0.1$ — **don't pick it**, it's on the edge of blowing up. Steepest descent is at LR $\approx10^{-2}$. **Good starting LR $=10^{-2}$** (≈3–10× below the minimum) — use it as `max_lr` for a one-cycle schedule (L5).

</div>

---

<!-- _class: section-divider -->

## Part 4 · Error analysis — the Ng move

---

# After training: look before you scale

Your model hits 82% val accuracy. The instinct is *bigger model / more epochs*. The high-value move is almost always different.

<div class="insight">

**Ng's rule.** Before adding complexity, **look at the errors.** Nearly always one failure category dominates — fixing *it* moves accuracy far more than architectural churn ever will. Manual inspection of ~100 mistakes is an afternoon; a wasted architecture search is a week.

</div>

---

# First, pick ONE single-number evaluation metric

You can't prioritize errors if you can't say which model is *better*. Ng: commit to **one** number before you start.

<div class="columns">
<div>

**Optimizing metric** — the one you maximize.
e.g. **F1** (balances precision & recall) on the dev set.

**Satisficing metric** — a bar to clear, not maximize.
e.g. *latency < 100 ms*, *model < 50 MB*.

</div>
<div>

| Model | F1 | Latency |
|---|---|---|
| A | 90 | 80 ms ✓ |
| B | **92** | 300 ms ✗ |
| C | 89 | 60 ms ✓ |

Pick **A**: best F1 *among* those that clear the latency bar. B is disqualified.

</div>
</div>

<div class="keypoint">

One optimizing metric + a few satisficing bars. Without it, every comparison becomes an argument.

</div>

---

# Error analysis: bucket, then prioritize

![w:900px](figures/lec03/svg/error_analysis_buckets.svg)

Sample ~100 mistakes, sort each into a failure category, and count. The tallest bar is your next experiment.

---

# The error-analysis worksheet

Ng's actual method is a **spreadsheet**. One row per mislabeled example, one column per hypothesized cause; tick all that apply, then tally.

| Example # | Blurry | Wrong label | Look-alike class | Small object |
|---|---|---|---|---|
| 12 | ✓ | | | |
| 37 | | | ✓ | |
| 58 | ✓ | | | ✓ |
| … | … | … | … | … |
| **% of errors** | **43%** | 8% | 31% | 18% |

<div class="keypoint">

**Blurry (43%)** is the dominant, fixable bucket → collect/augment blur. Chasing the 8% mislabeled bucket would cap your gain at 8% — the ceiling *is* the bucket size.

</div>

---

# From bucket to intervention

Don't stop at naming the error — convert each bucket into an **action**:

| Error bucket | Diagnosis | Intervention |
|---|---|---|
| blurry images | fails on motion blur | add blur augmentation / collect examples |
| rare class | too few examples | reweight loss / targeted collection |
| label ambiguity | humans disagree | clean labels / merge classes |
| background shortcut | uses spurious context | crop, mask, augment backgrounds |
| threshold error | probs fine, decision bad | tune the threshold on val |

<div class="insight">

Error analysis is not a post-mortem — it's the fastest way to decide **which experiment to run next.**

</div>

---

# Confusion matrix: which classes get mixed up

For classification, the confusion matrix *is* the error analysis, laid out as a grid. Off-diagonal mass shows *which* pairs the model confuses. On MNIST, the errors cluster in visually similar digits:

| True ↓ / Pred → | typical confusion | why |
|---|---|---|
| **4** | → **9** | open vs. closed top loop |
| **3** | → **5** | mirrored curves |
| **7** | → **1** | thin strokes, no crossbar |
| **8** | → **3** | left side occluded |

<div class="notebook">

**📓 Notebook · confusion matrix on MNIST** — train a small net, plot the confusion matrix, and pull the actual `4↔9` / `3↔5` misclassified images to eyeball them. *(ML ES 335 · `notebooks/confusion-mnist.ipynb`)*

</div>

---

# Ceiling analysis — for multi-stage pipelines

If your system is a pipeline (detect → crop → classify), which stage deserves the work? **Manually make one stage perfect**, measure the end-to-end gain, then move to the next.

![w:920px](figures/lec03/svg/ceiling_analysis.svg)

The stage whose "perfect version" lifts accuracy the most is where your effort pays off — the rest is polishing a component that isn't the bottleneck.

---

# Ablation discipline

When improving a model, change **one thing at a time** — otherwise you learn only that the final run was *different*, not *why*.

| Run | Change | Val | Interpretation |
|---|---|---|---|
| A | baseline | 82.0 | reference |
| B | stronger augmentation | 84.1 | likely useful |
| C | bigger model | 82.4 | small gain, more cost |
| D | aug + bigger model | 84.0 | the bigger model added ~nothing |

<div class="keypoint">

Keep the split fixed; log config + seed + commit + metric. If a gain is small, repeat with **3 seeds** — a 0.3% "improvement" is often just noise.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** You have **1000 misclassified validation images**. You can't fix everything. Describe the **procedure** that decides *what to fix first* — and say why you would **not** start by retraining a bigger model.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="math-box">

1. **Sample** ~100–200 of the 1000 at random (don't read all 1000).
2. **Tag** each into failure buckets in a spreadsheet — blur, rare class, wrong label, look-alike, background shortcut.
3. **Tally** the percentage in each bucket.
4. **Rank** by *frequency × fixability* — a big bucket you can actually address.
5. **Estimate the ceiling**: if that bucket vanished, how much accuracy would you gain? (bucket % is the cap.)
6. **Fix the top bucket**, retrain, re-measure — then repeat.

</div>

<div class="insight">

A bigger model is a blind, expensive guess — it may not touch the dominant bucket at all (e.g. it won't fix *mislabeled* data or a *background shortcut*). Two hours of looking beats a week of blind scaling.

</div>

---

<!-- _class: section-divider -->

## Part 5 · The data pipeline, splits & leakage

---

# Dataset → DataLoader → device

The pipeline feeds the GPU: a `Dataset` yields one example, a `DataLoader` batches and prefetches with worker processes, and the batch moves to the device.

![w:920px](figures/lec03/svg/dataloader_pipeline.svg)

Benchmark the loader **alone** before blaming the model — if data throughput is the bottleneck, a bigger model or better optimizer won't help.

---

# The batch contract

Before the model sees a batch, verify it satisfies a contract. Many "model bugs" are batch bugs.

| Item | Expectation (classification) | Common bug |
|---|---|---|
| `x.shape` | `[B, C, H, W]` or `[B, d]` | missing batch dim |
| `x.dtype` | `float32` / `bfloat16` | raw `uint8` pixels |
| `x` range | normalized, ~centered | still in `[0, 255]` |
| `y.shape` | `[B]` for hard labels | one-hot where index expected |
| `y.dtype` | `torch.long` | float class "index" |

```python
assert x.dtype == torch.float32 and y.dtype == torch.long
```

---

# train / val / test — three roles, three questions

Three splits because they answer three *different* questions — and mixing them corrupts the answer.

| Split | Question it answers | Touched by |
|---|---|---|
| **train** | what parameters fit the data? | the optimizer |
| **val** | which *hyperparameters* generalize? | you (tuning) |
| **test** | how will it do in the wild? | **once**, at the end |

<div class="warning">

Every time you tune on the val set you leak a little into it — so a held-out **test** set, looked at *once*, is the only honest estimate. When data is scarce, **k-fold cross-validation** (from ES 335) reuses the data for the val role without a fixed split.

</div>

---

# Leakage — the silent killer

Leakage makes the training recipe look *perfect* while the model has learned the wrong thing. The fix is almost always **how you split**.

| Problem | What leaks | Better split |
|---|---|---|
| same patient in train & val | patient-specific artifacts | split **by patient** |
| adjacent video frames | near-duplicate frames | split **by video/scene** |
| time series, random split | the future into the past | **chronological** split |
| repeated documents | near-duplicate text | split **by source** |
| normalize before splitting | test statistics into train | fit scaler on **train only** |

<div class="keypoint">

Check the split *before* celebrating a curve. A too-good val score is a leak until proven otherwise.

</div>

---

# Distribution shift is the next test

Even with no leakage, val may not match deployment.

| Shift | Example | Response |
|---|---|---|
| covariate | new camera / hospital | representative val/test |
| label | class priors change | report **per-class** metrics |
| concept | target definition drifts | refresh labels, monitor |
| temporal | user behavior evolves | **time-based** test set |

<div class="insight">

The question is not "did val improve?" It's **"does val measure the future cases I care about?"**

</div>

---

<!-- _class: section-divider -->

## Part 6 · Reproducibility

---

# Five layers of reproducibility

A result you can't reproduce is a result you can't build on. Five layers, cheapest first:

![w:900px](figures/lec03/svg/reproducibility_stack.svg)

The small habits here — a seed, a saved config, a git commit — are what save you *weeks* when a number looks wrong three months later.

---

# Seeds and determinism

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# only if bit-exact reruns matter (slower):
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

<div class="keypoint">

**Bitwise determinism ≠ statistical reproducibility.** The flags buy bit-exact reruns (and *even then* not across GPU / library versions). What you actually want is for the **conclusion to survive a seed change** — so run 3 seeds and report the spread.

</div>

---

# Save and load checkpoints correctly

<div class="columns">
<div>

**Save the `state_dict`** (+ optimizer, epoch, config):
```python
torch.save({
  'model': model.state_dict(),
  'optim': opt.state_dict(),
  'epoch': ep, 'config': cfg,
}, 'ckpt.pt')
```

</div>
<div>

**Load it back:**
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

Never `torch.save(model)` — it pickles the *class*, which breaks the moment you refactor. Always save the `state_dict`.

</div>

---

# The minimal experiment record

Every serious run should leave enough evidence to reproduce and compare it:

| Record | Why it matters |
|---|---|
| git commit | exact code |
| config file | hyperparameters & paths |
| data version / split id | same examples in each split |
| random seed | reproducibility + variance estimate |
| hardware + precision | speed & numeric behavior |
| best + final checkpoint | resume and deploy |

<div class="keypoint">

If you cannot compare two runs later, the experiment **did not really happen.**

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Read one signal — a gap, a flat loss, a dominant error bucket — and apply the single fix it points to. A checklist, not magic.**

</div>

- **Diagnose first**: the *train–val gap* says bias (add capacity) vs variance (add data / regularization).
- **Debug on a ladder**: *overfit one batch* before you ever touch the LR.
- **One knob**: the *LR finder* picks the learning rate in ~100 steps.
- **Look before you scale**: *error analysis* on 100 mistakes beats blind architecture search.
- **Trust the number**: guard against *leakage*, log the *seed*, keep the record.

**Next (L5–L6):** the *fixes* become the topic — momentum & schedules for the LR, weight decay & dropout for the variance.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Whole-lecture framing — error analysis, single-number metric, avoidable bias & human-level performance** — A. Ng, *Deep Learning Specialization*, **Course 3 · "Structuring Machine Learning Projects"** (the primary inspiration; his most practical, no-code course).
- **Bias–variance, cross-validation, confusion matrix on MNIST** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (notebooks `bias-variance.ipynb`, `confusion-mnist.ipynb`, `hyperparameter-optimisation.ipynb`).
- **The debugging ladder — overfit one batch, "become one with the data"** — A. Karpathy, *"A Recipe for Training Neural Networks"* (2019).
- **Learning-curve diagnosis interactive** — Interactive Lab · `~/git/interactive`.

Figures adapted from the ES 667 figure library (`figures/lec03/`). All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the bias–variance decomposition, derived

Let $y=f(x)+\varepsilon$ with $\mathbb E[\varepsilon]=0,\ \text{Var}(\varepsilon)=\sigma^2$, and $\hat f$ the model trained on a random dataset. Write $\bar f=\mathbb E[\hat f(x)]$. Add and subtract $\bar f$ inside the expected error:

$$\mathbb E\big[(y-\hat f)^2\big] = \mathbb E\big[(f+\varepsilon-\hat f)^2\big] = \underbrace{\sigma^2}_{\text{noise}} + \mathbb E\big[(f-\hat f)^2\big]$$

since $\varepsilon$ is independent with mean $0$ (the cross-term vanishes). Now split the second term with $\pm\bar f$:

$$\mathbb E\big[(f-\hat f)^2\big] = \underbrace{(f-\bar f)^2}_{\text{bias}^2} + \underbrace{\mathbb E\big[(\hat f-\bar f)^2\big]}_{\text{variance}}$$

again the cross-term is zero because $\mathbb E[\hat f-\bar f]=0$. **Total $=$ bias$^2 +$ variance $+ \sigma^2$** — three non-negative pieces, and $\sigma^2$ is the floor no model can beat. *(Full version: ML ES 335 `bias-variance.tex`.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why the LR sweep is geometric

The finder multiplies the LR by a constant $r$ each step, so after $k$ steps $\text{LR}_k=\text{start}\cdot r^{k}$ — i.e. it is **linear in $\log(\text{LR})$**. To span $[\text{start},\text{end}]$ in $n$ steps:

$$\text{end}=\text{start}\cdot r^{\,n-1}\;\Longrightarrow\; r=\Big(\tfrac{\text{end}}{\text{start}}\Big)^{1/(n-1)}$$

For $\text{start}=10^{-7},\ \text{end}=10,\ n=100$: $\;r=(10^{8})^{1/99}\approx1.20$. A *linear* sweep would waste almost all its steps in the flat, uninteresting region — the useful LRs span many **orders of magnitude**, so you sweep them multiplicatively and plot on log-x.

<div class="insight">

**Modern caveat.** The clean bias–variance U-curve can break for very large models: past the interpolation point test error can fall *again* (**double descent**). The train–val *gap* remains a valid signal, but "bigger always overfits" is not a law at scale. *(Belkin et al., 2019.)*

</div>
