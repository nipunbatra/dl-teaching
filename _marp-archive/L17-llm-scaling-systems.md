---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Large Language Models · Scaling & Systems

## Lecture 17 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

The architecture is **finished**. A decoder-only Transformer trained by cross-entropy on next-token prediction — the exact object you already built — runs the *identical* code path in GPT-2 and in a frontier model. One thing changed between a 2019 curiosity and a 2023 phenomenon: **scale.**

<div class="keypoint">

**This lecture is about turning one knob — compute — *well*.** How big a model, on how much data, and the few serving tricks that make it fit. Everything reduces to one accounting fact, $C \approx 6ND$, and one recipe, $D \approx 20N$.

</div>

$$\underbrace{C \approx 6ND}_{\text{the budget}} \;\rightarrow\; \underbrace{D \approx 20N}_{\text{Chinchilla split}} \;\rightarrow\; \underbrace{\text{RoPE · KV-cache · GQA}}_{\text{make it serve}} \;\rightarrow\; \underbrace{\text{emergence}}_{\text{what falls out}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Everything you already have:**
- **L01** — cross-entropy = NLL, the *only* LLM training loss
- **L13–L14** — the decoder-only Transformer, trained as a next-token predictor
- Today changes **none** of it — only its size

</div>
<div>

**Where today points:**
- **L18** — a model that *completes* text is not yet one that *helps* you; that gap is **alignment**
- The serving tricks here (KV-cache, GQA) are what make deployment affordable

</div>
</div>

<div class="insight">

One object, one loss, from ES 335 to the frontier. The only new variable is how much compute you pour in — so we spend the lecture learning to spend it.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Scaling laws & the $6ND$ budget

*Given a fixed compute budget, how big a model — on how much data?*

---

# The object never changes — only its size

![w:860px](figures/lec15/svg/next_token_pipeline.svg)

<div class="keypoint">

An LLM is *one* thing: a function that reads the tokens so far and outputs a probability for every possible next token. You sample one, glue it on, repeat. Scaling laws, RoPE, GQA, emergence — all of it is about making **this exact object** bigger and cheaper.

</div>

---

# What actually changed · 2018 → today

<div class="columns">
<div>

### Architecture

**Unchanged.** Still a decoder-only Transformer. A 2018 student would recognize a frontier model's high-level shape — though the norm, positional encoding, and attention details were all revamped (Parts 2–3).

</div>
<div>

### Scale (as of this writing)

Params · 117M → 1T+ ($10{,}000\times$)
Data · 1B → 15T tokens ($15{,}000\times$)
Compute · $10^{20}$ → $10^{25}$ FLOPs ($100{,}000\times$)
Context · 512 → 1M+ tokens

</div>
</div>

<div class="insight">

Sutton's *bitter lesson* — general methods that leverage computation beat hand-engineered ones. LLMs are Exhibit A: the wins came from scale, not cleverness.

</div>

---

# Two knobs, one budget

You have a **fixed compute budget** $C$ — the FLOPs you can afford. You spend it on exactly two knobs:

<div class="math-box">

- **$N$** — number of **parameters** (how big the model is)
- **$D$** — number of **training tokens** (how much data it sees)

$$\boxed{\,C \approx 6\,N\,D\,}$$

</div>

Turn $N$ up and you can afford fewer tokens $D$; turn $N$ down and you can pour in more data. **That single trade-off — how to split $C$ between $N$ and $D$ — is the entire scaling-laws question.**

---

# Where $6ND$ comes from — the FLOPs accounting

We own this rule here; every FLOPs estimate in the course rests on it. Count the multiply-adds a **single token** costs, then multiply by $D$ tokens.

<div class="math-box">

1. Each parameter sits in one matrix multiply → **1 multiply + 1 add = 2 FLOPs** per token (forward).
2. Forward pass over all params: $\;2N$ FLOPs / token.
3. Backward pass computes gradients w.r.t. inputs *and* weights → about $2\times$ the forward work: $\;4N$ FLOPs / token.
4. Forward + backward: $\;2N + 4N = 6N$ FLOPs / token.
5. Over $D$ training tokens: $\;\boxed{C \approx 6ND}$

</div>

*The "6" is one forward + two backward; it's an estimate (ignores attention's $O(T^2)$ term, which is small while $d \gg T$), but it's the number everyone plans with.*

---

# Where the FLOPs physically run · parallelism

$C \approx 6ND$ says *how much* compute; it hides *where* it runs. A 70B model's weights alone are 140 GB — far past one accelerator's memory. So a real training run is split across thousands of them, three orthogonal ways (usually stacked):

![w:560px](figures/lec15/svg/distributed_training.svg)

<div class="insight">

**Data** (replicate the model, split the batch), **tensor** (split each matrix across devices), **pipeline** (put different layers on different devices). Combine all three — "3D parallelism" — plus **ZeRO** optimizer-state sharding, and a 70B run lands on ~1000 GPUs. This is the "Systems" in the lecture title: $6ND$ is the bill; parallelism is *how you pay it*.

</div>

---

# The scaling law · in one picture

![w:760px](figures/lec15/svg/loss_vs_compute.svg)

<div class="insight">

Pour in more compute and the loss falls along a near-straight line on a **log-log** plot — a power law. So you can *predict* how good a model will be **before** you train it. That predictability is what turned scaling from a gamble into an engineering plan.

</div>

---

# The shape of the law · what each term means

Chinchilla's fit is not one number but a three-term law for the loss:

<div class="math-box">

$$L(N, D) = \underbrace{E}_{\text{irreducible}} + \underbrace{\frac{A}{N^{\alpha}}}_{\text{too few params}} + \underbrace{\frac{B}{D^{\beta}}}_{\text{too little data}}$$

</div>

- **$E$** — the entropy of language itself; no model, however large, beats it.
- **$A/N^\alpha$** — the penalty for a model too *small* to fit the patterns → vanishes as $N \to \infty$.
- **$B/D^\beta$** — the penalty for *too little data* → vanishes as $D \to \infty$.

<div class="insight">

Because both correction terms shrink as *powers*, they never hit zero — you buy loss reductions forever, just with diminishing returns. Minimizing this $L$ under the constraint $C = 6ND$ is exactly what produces the $D \approx 20N$ rule.

</div>

---

# Intuition · a straight line you can extrapolate

A power law is a **straight line on log-log axes** — and a straight line is the most forecastable object in science: measure two cheap points and you know where the expensive one lands.

<div class="keypoint">

This is why frontier labs run a **ladder** of small models first — 10M, 100M, 1B params — fit the line, then **extrapolate** to predict the loss of a run that will cost tens of millions of dollars, *before spending it*. Scaling turned "train it and pray" into "read it off the line."

</div>

<div class="insight">

The honest caveat: the line predicts **loss** (next-token cross-entropy), not **capability**. Loss falls smoothly and predictably; whether that smooth drop *buys* you arithmetic or reasoning is a separate, bumpier question — the subject of Part 4. Predictable fuel, less-predictable payoff.

</div>

---

# Explore it yourself

<div class="notebook">

**🎛 Interactive · KV-cache & serving cost** — vary layers, heads, context, and group size; watch the cache (and the serving bill) grow. We use it in Part 3 too. *(Interactive Lab · `interactive/articles/kv-cache`)*

</div>

<div class="notebook">

**📓 Notebook · scaling-laws sandbox** — fit $L(N,D) = E + A/N^\alpha + B/D^\beta$ to toy runs, then minimize under $C = 6ND$ and read off the compute-optimal $N$. *(ES 667 · `notebooks/17-scaling-laws.ipynb`)*

</div>

---

# The Chinchilla result

<div class="keypoint">

Hoffmann et al. 2022 · *Training Compute-Optimal Large Language Models.*

For a fixed budget $C$, loss is minimized when $N$ and $D$ scale **roughly proportionally**:

$$C \approx 6ND, \qquad \frac{D}{N} \approx \mathbf{20 \text{ tokens per parameter}}$$

</div>

*Fine print* · the paper fits $L(N,D) = E + A/N^\alpha + B/D^\beta$ and minimizes it under $C = 6ND$; "20 tokens per parameter" is the headline ratio that falls out, not the whole law.

---

# Intuition · balance the model and the data

Chinchilla in one sentence: **for a fixed budget, don't buy a giant engine you can't afford to fuel.** The sweet spot is roughly **20 training tokens per parameter** — model and data sized to match.

<div class="columns">
<div>

**Too big, too little data** ($D/N \ll 20$)
A huge model memorizing a thin dataset — a race-car engine with a coffee-cup fuel tank. GPT-3 sat here.

</div>
<div>

**Too small, too much data** ($D/N \gg 20$)
A tiny model drowning in tokens it lacks the capacity to absorb — a moped towing a fuel truck. Compute spent on data the model can't use.

</div>
</div>

<div class="insight">

Formally, the optimum **balances the two penalty terms** $A/N^\alpha$ and $B/D^\beta$: spend each marginal FLOP wherever it buys the most loss. At the compute-optimal point you are starving neither the model nor the data — which is exactly where $D \approx 20N$ comes from.

</div>

---

# Chinchilla · in one chart

![w:880px](figures/lec15/svg/chinchilla_scaling.svg)

<div class="insight">

The compute-optimal frontier is a *line*: as budget grows, the best model grows its parameters **and** its data together. Sitting off that line — too big for your tokens, or too small — wastes budget.

</div>

---

# GPT-3 was badly *undertrained*

![w:820px](figures/lec15/svg/chinchilla_sweet_spot.svg)

| | Params $N$ | Tokens $D$ | $D/N$ | Rule wants $D=20N$ |
|:-:|:-:|:-:|:-:|:-:|
| Chinchilla | 70B | 1.4T | **20** | ✓ on the line |
| **GPT-3 (2020)** | 175B | 300B | **≈ 1.7** | $20\times175\text{B}=$ **3.5T** |

GPT-3 saw 300B tokens where the rule asks for 3.5T — **~10× undertrained**. Chinchilla, ~2.5× *smaller*, beat it by spending the same budget on data instead of parameters.

---

# The 2023+ twist · deliberate *over*-training

Modern models often train *well past* Chinchilla optimal:

| Model | Params | Tokens | $D/N$ | Note |
|-------|--------|--------|-----|------|
| Chinchilla | 70B | 1.4T | 20 | training-compute optimal |
| A 70B, 2T-token model | 70B | 2T | 29 | slightly over |
| An 8B, 15T-token model | 8B | 15T | ~1875 | **wildly over** |

<div class="insight">

Chinchilla optimizes *training* compute. But a model earns its keep at **inference**. A smaller, over-trained model is cheaper *per query* — you repay the extra training compute many times over in serving. Serving economics, not training, drives 2024-era recipes.

</div>

---

# Intuition · over-train for cheaper inference

Chinchilla minimizes the **training** bill — paid once. But a deployed model pays an **inference** bill on *every* query, forever. So the modern move: deliberately shrink the model and over-feed it data.

<div class="keypoint">

**Pay more upfront to pay less per ride.** A smaller-but-over-trained model costs extra training FLOPs *once*, then is cheaper on every one of billions of future queries — and it fits on smaller, cheaper hardware. It sits *below* Chinchilla on training efficiency, but far *above* it on **total lifetime** efficiency.

</div>

<div class="insight">

The equation is still $C \approx 6ND$ — you have just stopped minimizing $C$. You now minimize *training + lifetime-inference* cost, and that objective pushes $N$ **down** and $D$ **up**. We put numbers on the trade in **Practice problem 4**.

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** You train a **7B-parameter** model on **1.4T tokens**. Estimate the total training compute in FLOPs using $C \approx 6ND$. Then compute $D/N$ — is this model under- or over-trained by Chinchilla's standard?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

$$C \approx 6ND = 6 \cdot (7\times10^9) \cdot (1.4\times10^{12}) = 6 \cdot 9.8\times10^{21} \approx \mathbf{5.9\times10^{22}\ \text{FLOPs}}$$

$$\frac{D}{N} = \frac{1.4\times10^{12}}{7\times10^9} = \mathbf{200} \;\gg\; 20$$

<div class="keypoint">

**$10\times$ over-trained** — exactly the modern serving-first regime. Sanity check on hardware: at ~$4\times10^{14}$ effective FLOP/s per accelerator, that's $\sim1.5\times10^8$ GPU-seconds $\approx$ **1700 GPU-days** — a few days on ~1000 accelerators.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** You are handed a compute budget of $C = 1.2\times10^{24}$ FLOPs. Using $C \approx 6ND$ **and** the Chinchilla split $D \approx 20N$, solve for the compute-optimal $N$ and $D$.

</div>

*Try it before the next slide. (Hint: substitute $D=20N$ into $C=6ND$ to get one equation in $N$.)*

---

# Solution · practice problem 2

Substitute $D = 20N$ into $C = 6ND$:

$$C = 6N(20N) = 120\,N^2 \;\Longrightarrow\; N = \sqrt{\frac{C}{120}} = \sqrt{\frac{1.2\times10^{24}}{120}} = \sqrt{10^{22}} = \mathbf{10^{11}}\ (100\text{B})$$

$$D = 20N = 2\times10^{12} = \mathbf{2\text{T tokens}} \qquad \text{check: } 6\cdot10^{11}\cdot2\times10^{12} = 1.2\times10^{24}\ \checkmark$$

<div class="insight">

Note the model is **not** simply $10\times$ bigger when budget grows $10\times$ — both $N$ and $D$ scale as $\sqrt{C}$ (Appendix). New compute is spread across *both* axes.

</div>

---

# Worked numeric · pricing a frontier run

Scaling laws turn a training run into a *budget line item*. Take a frontier-scale compute of $C \approx 2\times10^{25}$ FLOPs (illustrative, as of this writing) and price it out.

<div class="math-box">

- Each accelerator delivers $\sim\!4\times10^{14}$ **effective** FLOP/s (fp16, real utilization — well below peak).
- Accelerator-seconds: $\dfrac{2\times10^{25}}{4\times10^{14}} = 5\times10^{10}\ \text{s} \approx \mathbf{1600}$ **accelerator-years**.
- Run on 10,000 of them: $\approx \mathbf{58}$ **days** wall-clock.
- At ~\$2 per accelerator-hour: $5\times10^{10}\,\text{s} = 1.4\times10^{7}$ accelerator-hours $\Rightarrow$ **~\$28 million**.

</div>

<div class="insight">

The exact figures age fast; the *chain* does not: $C \approx 6ND \rightarrow$ FLOPs $\rightarrow$ accelerator-hours $\rightarrow$ dollars is the back-of-envelope behind every training proposal. It is also *why* predicting the loss before you spend (the scaling law) is worth so much — you are committing tens of millions on that one forecast.

</div>

---

<!-- _class: section-divider -->

## Part 2 · Capabilities that emerge from scale

*New behaviors appear — with no new training objective.*

---

# In-context learning · the most surprising capability

At pretraining the model learns *only* next-token prediction. Yet past ~50B parameters it starts to **learn at inference time** from examples in the prompt:

```
Translate to French:
sea otter → loutre de mer
cheese    → fromage
banana    → banane
carrot    → ???
```

Given three examples, the model infers the pattern (English → French) and emits "carotte" — with **no weight updates**. It was never *told* the task.

<div class="keypoint">

**Few-shot learning without gradient steps.** Emergent at scale, and the foundation of all modern prompting.

</div>

---

# Intuition · the prompt is a training set that vanishes

Ordinary learning: show examples, take gradient steps, *change the weights.* In-context learning does the first and skips the rest — the examples live only in the prompt, and when the prompt ends the "learning" evaporates. No weight ever moved.

<div class="keypoint">

So *where* did the learning go? Into **pretraining**. Predicting the next token across the whole internet forced the model to build a general pattern-completer — one that, at inference, reads "sea otter → loutre; cheese → fromage; carrot →" as a pattern to finish. The task-learning was **amortized into the weights once**; the prompt just *selects* which learned skill to run.

</div>

<div class="insight">

This reframes prompting: you are not teaching the model, you are **addressing** a capability it already has. Same weights, a different program — chosen by words. (That is the exact question the discuss box a few slides on asks you to answer.)

</div>

---

# Prompting is the interface

The same frozen weights expose three modes, chosen purely by how you *write* the prompt:

<div class="columns">
<div>

- **Zero-shot** — state the task, no examples: *"Translate to French: carrot →"*
- **Few-shot** — prepend $k$ worked examples (in-context learning).
- **Instruction** — a natural-language spec: *"Summarize this in one line."*

</div>
<div>

<div class="insight">

No retraining, no new parameters — the prompt *is* the program. This is why "prompt engineering" exists at all, and why behavior is so sensitive to phrasing.

</div>

</div>
</div>

More examples generally help — until the pattern is clear or the context window fills. That context wall is exactly what Part 3's serving tricks exist to push back.

---

# Chain of thought · "let's think step by step"

Ask the model to *show its work* and multi-step accuracy jumps:

```
Q: Roger has 5 tennis balls. He buys 2 cans of 3 balls each.
   How many balls does he have?

Without CoT:  "11."            ← often wrong at small scale
With CoT:     "Start with 5. 2 × 3 = 6 more. 5 + 6 = 11."  ← reliable
```

CoT *emerges* at scale — at ~10B params, "think step by step" does nothing; past ~60B it adds 20+ points on math benchmarks.

<div class="insight">

The prompt itself is a control knob — no fine-tuning needed. Spending more tokens *thinking* buys accuracy. This thread becomes today's **reasoning models**.

</div>

---

# Reasoning models · spend compute at inference

The latest generation (as of this writing) explicitly trains the model to produce a long internal chain of thought *before* answering:

- Trained with RL over the reasoning trace (process-style rewards).
- Spends $10\times$–$100\times$ more compute **per answer**.
- Often dramatically better on math, code, and logic.

<div class="keypoint">

The lesson repeats one level up: after *training* compute, **inference** compute is the next scaling axis. We meet the RL machinery that trains these traces in **L18**.

</div>

---

# See it live, then a question

<div class="notebook">

**🎛 Interactive · in-context learning** — watch few-shot prompting solve a task the model was never trained on, example by example. *(Interactive Lab · `interactive/articles/in-context-learning`)*

</div>

<div class="notebook">

**🎛 Interactive · softmax temperature** — the sampling knob behind "creative vs deterministic" generation; low $T$ sharpens, high $T$ flattens the next-token distribution. *(Interactive Lab · `interactive/articles/softmax-temperature`)*

</div>

<div class="popquiz">

**Discuss.** In-context learning does few-shot learning with **zero weight updates**. If the weights don't change, *where* is the "learning" happening — and what does that say about what pretraining actually built?

</div>

---

<!-- _class: section-divider -->

## Part 3 · Serving-shaped design

*Why does generation run out of memory before it runs out of compute?*

---

# Serving is memory-bound, not compute-bound

Training is one giant compute job. **Serving** is millions of short ones — and the wall you hit is **memory**, because every token you generate must attend to *every* token before it.

<div class="keypoint">

Three design choices exist almost entirely to make serving cheap:
- **RoPE** — relative positions, so context can extend far past the training length.
- **The KV-cache** — store past keys/values instead of recomputing them.
- **MQA / GQA** — shrink that cache so it fits on the GPU.

</div>

The rest of Part 3 is these three, in order.

---

# The problem with absolute positional encoding

Sinusoidal and learned PE inject position by **adding** a vector to the token embedding:

$$x_\text{pos}(t) = \text{emb}(\text{token}) + \text{pe}(t)$$

Two failures:

- **Absolute only** — the model can't easily learn "2 positions apart" as a reusable primitive.
- **Extrapolation breaks** — trained on ≤ 4k tokens, it falls apart at 10k. Bad for long-context serving.

---

# RoPE · the spinning-pointer intuition

![w:840px](figures/lec15/svg/rope_rotation.svg)

<div class="insight">

Picture $Q$ and $K$ as pointers on a clock. Instead of *adding* a position vector, **rotate** each pointer by an angle set by its position — position 1 → 10°, position 3 → 30°. The attention score is a dot product, which depends on the **angle between them**, $30°-10°=20°$ — exactly the offset $n-m=2$. The model reads **relative position directly.**

</div>

---

# RoPE · three properties that matter for serving

<div class="math-box">

1. **Relative positions, for free** — after rotation the score depends only on the offset $n-m$, not on where each token sits.
2. **Extrapolates beyond training length** — frequencies are fixed, so a model trained at 4k context stretches to 32k+ with minor fixes.
3. **Zero added parameters** — rotations are deterministic given position; no `nn.Embedding(max_len, d)` to allocate.

</div>

| Year | Frontier context | Enabled by |
|:-:|:-:|:-:|
| 2020 | ~2k | — |
| 2023 | 32k–100k | RoPE + FlashAttention |
| today | 1M–10M (as of this writing) | RoPE extrapolation + smaller KV-cache |

<div class="notebook">

**🎛 Interactive · positional encoding** — compare sinusoidal vs rotary; see why only RoPE's score is a function of the offset. *(Interactive Lab · `interactive/articles/positional-encoding`)*

</div>

---

# Two phases of serving · prefill vs decode

Generation splits into two very different regimes — and only the second is the problem:

<div class="columns">
<div>

**Prefill** — read the whole prompt at once.
- All $T$ prompt tokens in **parallel**.
- Compute-bound (big matmuls), efficient.
- Happens **once** per request.

</div>
<div>

**Decode** — emit one token at a time.
- One token per step, **serially**.
- Each step re-attends to *all* prior tokens.
- Happens **thousands** of times.

</div>
</div>

<div class="insight">

Decode is where serving cost lives — and it would recompute every past key and value at every step. Caching those vectors is the fix; the cache's *size* becomes the new wall. That's the next slide.

</div>

---

# The KV-cache · what generation actually stores

<div class="keypoint">

To generate token #1001 the model must attend to the 1000 tokens before it. Rather than recompute their keys and values every step, it **stores them in a table — the KV-cache** — holding one $K$ and one $V$ vector for *every past token × every layer × every head.*

</div>

The table grows with context until **it**, not compute, fills the GPU. State the object first; then we'll size it — and that size is exactly what GQA exists to shrink.

---

# KV-cache · build the size, layer by layer

<div class="math-box">

1. **One token, one head, one layer** — store $K$ and $V$, each $d_h$ numbers → $2 d_h$.
2. **All $H$ heads** → $2 H d_h$.
3. **All $L$ layers** → $2 L H d_h$.
4. **All $T$ tokens, batch $B$** → $2 L H d_h \, T B$ numbers.
5. **In bytes (fp16, 2 bytes each):**

$$\text{KV-cache} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot B \cdot 2 \ \text{bytes}$$

</div>

The leading 2 is "$K$ and $V$"; the trailing 2 is "bytes per fp16 number."

---

# Worked numeric · the 84 GB wall

Setup: $L=80,\ H=64,\ d_h=128,\ T=32{,}000,\ B=1$, fp16.

$$\text{Cache} = 2 \cdot 80 \cdot 64 \cdot 128 \cdot 32{,}000 \cdot 1 \cdot 2 \approx 8.4\times10^{10}\ \text{bytes} = \mathbf{84\ \text{GB}}$$

<div class="insight">

The 70B weights (fp16) take $70\times10^9 \cdot 2 = 140$ GB. The KV-cache for **one** 32k sequence adds another **84 GB** — more than half the model again, and it scales with batch size. This is *the* long-context serving bottleneck.

</div>

---

# MHA → MQA → GQA · share the keys and values

![w:840px](figures/lec15/svg/gqa_variants.svg)

Only the **key/value** heads cost cache memory; query heads don't. So keep many query heads for expressiveness, but let groups of them **share** one K/V head.

---

# GQA · the shared-notebook analogy

<div class="insight">

The KV-cache is the model's **notebook**.
- **MHA** — 64 students each keep a private notebook → biggest cache.
- **GQA** — 8 study groups share one notebook each → $8\times$ smaller, almost no quality drop.
- **MQA** — all 64 share one notebook → smallest, but they overwrite each other (quality drops).

</div>

![w:700px](figures/lec15/svg/kv_cache_bars.svg)

---

# How GQA shrinks the cache · $84 \to 10.5$ GB

Refine the formula with $H_q$ query heads and $H_{kv}$ key/value heads — only $H_{kv}$ enters:

$$\text{Cache} = 2 \cdot L \cdot H_{kv} \cdot d_h \cdot T \cdot B \cdot 2$$

| Variant | $H_{kv}$ | Cache (70B, $T=32k$) |
|---------|:-:|:-:|
| **MHA** | 64 | $\approx$ **84 GB** |
| **GQA** (8 groups) | 8 | $\approx$ **10.5 GB** |
| **MQA** | 1 | $\approx$ **1.3 GB** (quality drops) |

<div class="keypoint">

GQA cuts the cache by $H_q/H_{kv} = 64/8 = \mathbf{8\times}$ with negligible quality loss — the modern default (e.g. `n_heads=64, n_kv=8`). Same model, one-eighth the serving memory.

</div>

---

# Beyond GQA · quantize the weights *and* the cache

GQA shrinks the cache by *sharing* K/V heads. The other lever is **precision** — store each number in fewer bits.

<div class="columns">
<div>

**Weights** · 70B in fp16 = 140 GB. In **int8**, 70 GB; in **int4**, ~35 GB — the whole model on a single accelerator, with small quality loss.

</div>
<div>

**KV-cache** · the same fp16 → int8 halving applies to every stored $K$ and $V$. Stack it on GQA and Part 3's 84 GB cache drops toward a few GB.

</div>
</div>

<div class="insight">

Quantization and GQA are **orthogonal** — they *multiply*. Modern serving stacks combine int8/int4 weights, GQA, and paged KV-caches to fit long contexts and big batches on commodity GPUs. Cheaper serving is won bit by bit.

</div>

<div class="notebook">

**🎛 Interactive · quantize & prune** — sweep bit-width and sparsity; watch model size fall and accuracy hold, then break. *(Interactive Lab · `interactive/articles/quantize-prune`)*

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** A model has $L=32$ layers, $H=32$ heads, $d_h=128$, context $T=8192$, batch $B=1$, fp16. **(a)** Compute the MHA KV-cache in GB. **(b)** With **GQA using 4 key/value groups**, what is the new size — and the factor saved?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

**(a) MHA:**

$$2 \cdot 32 \cdot 32 \cdot 128 \cdot 8192 \cdot 1 \cdot 2 = 2 \cdot 32 \cdot 32 \cdot 128 \cdot 8192 \cdot 2 \approx 4.3\times10^{9}\ \text{bytes} \approx \mathbf{4.3\ \text{GB}}$$

**(b) GQA, $H_{kv}=4$:** only the head count changes, $32 \to 4$, a factor of **8**:

$$\frac{4.3\ \text{GB}}{8} \approx \mathbf{0.54\ \text{GB}} \qquad \text{(saved } H_q/H_{kv} = 32/4 = 8\times\text{)}$$

<div class="insight">

The cache scales *linearly* in $H_{kv}$, so the saving is exactly the group ratio — independent of $L$, $T$, or $d_h$. That is why GQA is a free lunch for serving.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** A frontier lab ships an **8B-parameter** model trained on **~15T tokens** (Llama-3-scale, as of this writing). **(a)** Find $D/N$ and the training FLOPs. **(b)** A *compute-optimal* 8B model would see only $D = 20N$ tokens — how many times **more** training compute did over-training spend? **(c)** In one line: what does that extra spend buy?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

**(a)** $\;D/N = \dfrac{15\times10^{12}}{8\times10^{9}} \approx \mathbf{1875}$ tokens/param (vs Chinchilla's 20).

$$C = 6ND = 6\cdot(8\times10^9)\cdot(15\times10^{12}) \approx \mathbf{7.2\times10^{23}\ \text{FLOPs}}$$

**(b)** A compute-optimal 8B sees $D_{\text{opt}} = 20N = 160\text{B}$, so $C_{\text{opt}} = 6\cdot8\times10^9\cdot160\times10^9 \approx 7.7\times10^{21}$:

$$\frac{C}{C_{\text{opt}}} = \frac{7.2\times10^{23}}{7.7\times10^{21}} \approx \mathbf{94\times} \qquad\Big(=\tfrac{1875}{20}\Big)$$

<div class="keypoint">

**(c)** The model is still **8B**, so inference stays cheap ($\sim 2N$ FLOPs/token) — but its capability climbs far above a compute-optimally-trained 8B. Spending that same $7.2\times10^{23}$ budget *optimally* would instead build a $\sqrt{C/120}\approx\mathbf{77B}$ model — **~10× costlier to serve**. Over-training pays ~94× training compute **once** to serve a small model that punches like a large one.

</div>

---

<!-- _class: section-divider -->

## Part 4 · Emergent abilities — honestly

*Do bigger models just get better, or do they get different?*

---

# Why "emergence" looks surprising

**Null hypothesis · smooth scaling.** Bigger model → smoothly lower loss. A 100B is a bit better than a 10B. Intuitive — and *mostly true*.

The surprise: some complex tasks stay near-random until a threshold, then take off. **Why?** A multi-step task is a *product* of per-step accuracies:

<div class="math-box">

- Small model · 50% per step → 3-step accuracy $= 0.5^3 = 12.5\%$ (barely above chance).
- Larger model · 90% per step → 3-step accuracy $= 0.9^3 = 72.9\%$ (competent).

</div>

**Smooth** improvement in per-step accuracy becomes a **discontinuous-looking** jump in end-to-end success.

---

# Emergent abilities · the curves

![w:860px](figures/lec15/svg/emergent_abilities.svg)

<div class="insight">

On a hard exact-match metric, performance hugs the floor, then shoots up past a scale threshold. Wei et al. 2022 catalogued dozens of these — arithmetic, in-context learning, chain of thought — none trained for specifically.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** A task takes **5** reasoning steps and is marked correct only if **all 5** succeed (exact match). Model A gets each step right with probability $0.6$; Model B, $0.9$. **(a)** End-to-end accuracy of each? **(b)** Per step B is only $1.5\times$ better — what is the *end-to-end* ratio, and why does that make emergence look like a cliff?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

**(a)** All-or-nothing over 5 independent steps means the accuracies **multiply**:

$$\text{A}: 0.6^5 \approx \mathbf{7.8\%} \qquad \text{B}: 0.9^5 \approx \mathbf{59\%}$$

**(b)** End-to-end ratio $\approx 59/7.8 \approx \mathbf{7.6\times}$ — a $1.5\times$ per-step edge compounds into a $7.6\times$ task-level gap.

<div class="insight">

Because success is a **product** of per-step accuracies, a *smooth* per-step improvement stays pinned near the floor on an exact-match metric, then lifts off once per-step accuracy clears a threshold. That is precisely Schaeffer's point (next slide): the **capability** rose gradually; the **metric** manufactured the cliff. Score by per-step log-prob instead and the "emergence" softens into a ramp.

</div>

---

# The controversy · metric artifact vs real

<div class="columns">
<div>

### Emergentists

- Genuine discontinuous jumps with scale.
- New qualitative behaviors (reasoning, tool use).
- Smaller models *can't* do these at all.

</div>
<div>

### Skeptics (Schaeffer 2023)

- Many curves are **metric artifacts**.
- Swap exact-match for a *smooth* metric (per-token log-prob) and the curve becomes gradual.
- The capability gap is real — but continuous.

</div>
</div>

<div class="keypoint">

Both are partly right. Capability improves **continuously** in log-probability; certain *thresholded* (match-or-fail) tasks look **discontinuous**. The user experience is still one of qualitative leaps — but the honest claim is "smooth underneath, sharp at the metric."

</div>

---

# Where abilities emerge — and where we go next

| Ability | Roughly emerges around |
|---------|:-:|
| Multi-digit arithmetic | ~13B |
| Basic code generation | ~13B |
| Few-shot in-context learning | ~50B |
| Chain-of-thought reasoning | ~60B |
| Tool use (with prompting) | ~70B+ |

<div class="insight">

These capabilities are latent in a raw pretrained model — but a next-token predictor is not yet *helpful, honest, or harmless.* Turning raw capability into a usable assistant is **alignment** — SFT, RLHF, DPO — which is **L18**, next.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**LLMs are the Transformer you already built, scaled — $C \approx 6ND$ sets the budget, $D \approx 20N$ spends it well, and RoPE + KV-cache + GQA make it serve.**

</div>

- **Scaling** — loss is a power law in compute; $C \approx 6ND$ (own it), Chinchilla's $D/N \approx 20$; modern models over-train for cheap inference.
- **Systems** — that budget physically runs on ~1000 GPUs via data / tensor / pipeline parallelism + ZeRO sharding.
- **Capabilities** — in-context learning, chain of thought, reasoning all *emerge* from scale, with no new objective.
- **Serving** — RoPE gives long context for free; the KV-cache is the memory wall; GQA cuts it $8\times$ ($84 \to 10.5$ GB), quantization multiplies the saving.
- **Emergence** — real capability gap, partly a metric artifact — smooth underneath, sharp at the exact-match metric.

**Next (L18):** from a text *completer* to a *helpful* assistant — alignment.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's ES 667 LLM notes (IIT Gandhinagar) and standard references:

- **Scaling laws (power law in compute)** — Kaplan et al. 2020, *Scaling Laws for Neural Language Models.*
- **Compute-optimal training, $C \approx 6ND$, $D/N \approx 20$** — Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models* (Chinchilla).
- **RoPE** — Su et al. 2021, *RoFormer: Rotary Position Embedding.*
- **GQA** — Ainslie et al. 2023, *GQA: Training Generalized Multi-Query Transformer Models.*
- **Emergent abilities & chain of thought** — Wei et al. 2022 (emergence; CoT); Schaeffer et al. 2023 (metric-artifact critique).
- **Distributed training (data / tensor / pipeline parallelism, ZeRO)** — Shoeybi et al. 2019 (Megatron-LM); Rajbhandari et al. 2020 (ZeRO); Zhao et al. 2023 (PyTorch FSDP).
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (scaling introduced in-line).

Figures reused from the ES 667 `figures/lec15/` library. Interactive explainers: `interactive/articles/{kv-cache, in-context-learning, positional-encoding, softmax-temperature, quantize-prune}`. All source materials © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why both $N$ and $D$ grow as $\sqrt{C}$

<div class="insight">

**Gist.** When compute goes up $10\times$, make the model about $\sqrt{10}\approx3.2\times$ bigger **and** train on $\sqrt{10}\times$ more data — *not* $10\times$ bigger. Spread new budget across **both** axes.

</div>

Combine the budget $C = 6ND$ with the Chinchilla split $D = 20N$:

$$C = 6N(20N) = 120\,N^2 \;\Longrightarrow\; N = \sqrt{\frac{C}{120}}, \qquad D = 20N = \sqrt{\frac{C}{0.3}}$$

Both optimal $N$ and optimal $D$ scale as $\sqrt{C}$. *Worked:* $C = 1.2\times10^{24} \Rightarrow N = 10^{11}$ (100B), $D = 2\times10^{12}$ (2T) — the Practice-problem-2 answer, now as a scaling law rather than a single point.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · RoPE · only the offset survives

Rotate the query at position $m$ and the key at position $n$ by $\theta_m = m\Theta$, $\theta_n = n\Theta$, using $R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$, so $q'_m = R_{\theta_m}q$ and $k'_n = R_{\theta_n}k$.

Two facts about rotations — $R_\theta^\top = R_{-\theta}$ and $R_{-\theta_m}R_{\theta_n} = R_{\theta_n - \theta_m}$ — collapse the attention score:

$$(q'_m)^\top k'_n = q^\top R_{\theta_m}^\top R_{\theta_n}\, k = q^\top R_{(n-m)\Theta}\, k$$

<div class="keypoint">

**The score depends only on the offset $n-m$, never on absolute positions** — relative position, zero extra parameters. In high dimensions, group coordinates into pairs and rotate each pair at its own frequency $\Theta_i$ (block-diagonal), so different pairs encode different position scales.

</div>

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · 3D parallelism & ZeRO

A frontier run stacks three *orthogonal* splits, then shards the optimizer state on top:

![w:470px](figures/lec15/svg/distributed_3d.svg)

<div class="keypoint">

**Data** (split the batch), **tensor** (split each matrix), **pipeline** (split the layers) — combine for "3D parallelism." Then **ZeRO / FSDP** shards the optimizer states (Adam's $m, v$ — two extra copies of *every* parameter) across the data-parallel replicas, so no single GPU holds the full $3\times$ memory. A 70B run is typically $\approx$ 4-way TP · 8-way PP · 32-way DP · ZeRO $\approx$ **1024 GPUs** — the systems substrate underneath $C \approx 6ND$.

</div>
