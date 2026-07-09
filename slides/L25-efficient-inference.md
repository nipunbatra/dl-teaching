---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Efficient Training & Inference

## Lecture 25 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A modern GPU can do about $10^{14}$ arithmetic operations per second — but read only about $10^{12}$ **bytes** per second from its memory. So the arithmetic is almost never the bottleneck: **waiting for bytes to arrive is.**

<div class="keypoint">

**Deep learning at scale is memory-bound, not compute-bound.**
Whether you are *training* a 70B model or *serving* it, the whole game is the same two moves: **move fewer bytes**, and **never redo work you've already done.** Every trick today — mixed precision, KV-cache, quantization, FlashAttention, speculative decoding — is one of those two moves.

</div>

You built the Transformer (L14–L15) and scaled it into an LLM (L16–L17). Those lectures asked *"can the model learn the function?"* Today asks the systems question they left open: **can we afford to train and run it?**

$$\underbrace{\text{mixed precision · accumulation · parallelism}}_{\text{fit the training}} \;\longrightarrow\; \underbrace{\text{KV-cache · quantization · Flash · speculative}}_{\text{cheapen the serving}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Builds directly on:**
- **L08** — the training loop, dtypes, batch size
- **L14–L15** — attention & the Transformer; **GQA** lives here
- **L17** — LLM scaling & systems (this lecture is its serving half)
- **L01** — the softmax / KL you'll see again in distillation

</div>
<div>

**Why "beyond Ng":**
Ng's specialization stops at *training a model well*. Deployment — quantization, KV-cache, FlashAttention, serving throughput — is the part every real system needs but few courses teach. His **philosophy** still guides us: *build the simplest thing that works, measure the bottleneck, then optimize only that.*

</div>
</div>

<div class="insight">

One spine runs through all seven techniques: **find the byte you're re-reading, and stop re-reading it.**

</div>

---

<!-- _class: section-divider -->

## Part 1 · Training at scale — fitting the model on the hardware

---

# The memory wall shows up in training first

Before you can serve a model you have to *fit the training step* in GPU memory. For each parameter you store, at minimum:

<div class="math-box">

**weights** (2–4 B) $+$ **gradients** (2–4 B) $+$ **optimizer state** (Adam: 8 B for two moments) $\approx$ **16 bytes / parameter**

</div>

A 7B model in FP32 Adam needs $\approx 7\text{B} \times 16 = 112$ GB — **more than one 80 GB GPU holds**, before a single activation. Two cheap fixes buy most of it back: **use fewer bytes per number** (mixed precision) and **shrink the batch you hold at once** (gradient accumulation).

---

# Mixed precision — compute in BF16, keep an FP32 master

<div class="insight">

**The analogy.** Do your rough arithmetic on a cheap notepad (16-bit), but keep the *official ledger* in full precision (32-bit). Fast scratch work, no drift in the running total.

</div>

Store and matmul the weights in **BF16** (16 bits); keep a **master copy of the weights in FP32** for the optimizer update, so tiny gradient steps don't vanish into rounding.

| Format | Bits | Exponent / Mantissa | Note |
|:--|:-:|:-:|:--|
| FP32 | 32 | 8 / 23 | the reliable master copy |
| FP16 | 16 | 5 / 10 | small range → needs *loss scaling* |
| **BF16** | 16 | **8** / 7 | same range as FP32 → the modern default |

BF16 keeps FP32's *exponent*, so it rarely overflows — that's why it beat FP16 for training. Net: **~2× less memory and bandwidth, ~same accuracy.** In PyTorch: `torch.autocast` + `GradScaler`.

---

# Gradient accumulation — simulate a big batch on a small GPU

Large batches train stably, but a big batch may not fit. **Gradient accumulation** splits it: run several small **micro-batches**, *sum* their gradients, and step the optimizer only once.

<div class="math-box">

$$\text{effective batch} = (\text{micro-batch size}) \times (\text{accumulation steps}) \times (\text{data-parallel GPUs})$$

</div>

The gradient of a sum is the sum of gradients, so accumulating $k$ micro-batches gives the *exact same* update as one batch $k\times$ larger — you just trade wall-clock time for memory. Only call `optimizer.step()` / `zero_grad()` once per $k$ micro-batches.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** You want an **effective batch of 512**, but your GPU only fits a **micro-batch of 32**.
(a) How many accumulation steps do you need on one GPU?
(b) Now you have **4 GPUs** running data-parallel. How many accumulation steps per GPU?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

<div class="columns">
<div>

**(a) One GPU**
$$\text{steps} = \frac{512}{32} = \mathbf{16}$$
Accumulate 16 micro-batches, then one optimizer step.

</div>
<div>

**(b) Four data-parallel GPUs**
Each GPU handles $512/4 = 128$ samples per step:
$$\text{steps} = \frac{128}{32} = \mathbf{4}$$

</div>
</div>

<div class="keypoint">

Effective batch $= 32 \times 4 \times 4 = 512$. **Data parallelism and accumulation are interchangeable ways to grow the batch** — parallelism spends *more GPUs*, accumulation spends *more time*.

</div>

---

# When one GPU isn't enough — three ways to split

If the model itself won't fit, split the *work* across devices:

| Parallelism | What's split | The cost |
|:--|:--|:--|
| **Data** | the *batch* (each GPU holds a full model copy) | all-reduce gradients every step |
| **Tensor** | each *layer's matmul* across GPUs | heavy per-layer communication → same node |
| **Pipeline** | *blocks of layers* across GPUs | "bubble" idle time at the ends |

<div class="insight">

Real 70B+ training combines all three (**3-D parallelism**). Notice the recurring theme: every scheme is a different answer to *"which bytes do we move between GPUs, and how often?"* — memory-bound thinking again.

</div>

---

# Data parallelism — replicate, then average

The simplest split: give every GPU a **full copy** of the model and a **different slice of the batch**. Each computes its own gradient; an **all-reduce** then averages them so every replica takes the *identical* step.

<div class="math-box">

$$\text{effective batch} = (\text{per-GPU batch}) \times (\text{\#GPUs}), \qquad \text{comm} = \text{one all-reduce of the full gradient, per step}$$

</div>

<div class="insight">

Plain data parallelism grows *throughput* but not *capacity* — every GPU still stores the whole model, gradients, and optimizer state. **FSDP / ZeRO** fix that: they **shard** those three across the group and rebuild each layer's weights on the fly, so the same replicas *also* cut per-GPU memory.

</div>

What it splits: **the data.** What it doesn't: the model — so it only helps once the model already fits on one GPU.

---

# Tensor parallelism — split one layer's matmul

When a single layer is itself too big, split the **matmul across GPUs**. For $Y = XW$, give each GPU a column-block of $W$; each computes a slice of $Y$, and an **all-reduce** stitches the slices back together.

<div class="math-box">

Megatron pattern: a **column-parallel** matmul feeding a **row-parallel** matmul ⇒ just **2 all-reduces forward** (+2 backward) per transformer block.

</div>

<div class="insight">

The catch: that all-reduce sits *inside every layer*, so tensor parallelism moves a lot of bytes very often. It is kept **within one node** on fast NVLink — spread it across nodes and the slow network stalls the whole model.

</div>

What it splits: **each layer's parameters and its matmul FLOPs** — cutting both memory and compute per GPU, at the price of constant communication.

---

# Pipeline parallelism — split the layers into stages

Cut the model **by depth**: GPU 0 holds layers 1–20, GPU 1 holds 21–40, and so on. Activations flow forward stage → stage, gradients flow back — and only the **boundary activations** cross GPUs, so the traffic is light.

The catch is the **bubble**: while stage 0 works, later stages sit idle waiting for input. You fill it by pushing many **micro-batches** through the pipe at once (GPipe / 1F1B schedules).

<div class="math-box">

$$\text{bubble fraction} \approx \frac{S-1}{m + S - 1} \qquad (S \text{ stages}, \; m \text{ micro-batches})$$

$S=4,\; m=1 \Rightarrow \tfrac{3}{4}$ idle; &nbsp; but $m=16 \Rightarrow \tfrac{3}{19} \approx 16\%$ idle.

</div>

<div class="keypoint">

**3-D parallelism** = data × tensor × pipeline: split the batch *across nodes*, the layers *into stages*, and each stage's matmuls *within a node*. That combination is what trains 70B+ models.

</div>

---

<!-- _class: section-divider -->

## Part 2 · Why serving is memory-bound — prefill vs decode

---

# We built it — now we must serve it

You trained the model *once*. Now **inference** re-runs all those frozen weights on *every* request, for *every* user. One request has two very different phases:

<div class="keypoint">

Think of it like **reading a chapter, then writing a summary**.

**Prefill** = reading the whole prompt at once — every token processed *in parallel*. **Compute-bound.**

**Decode** = writing the answer one token at a time — each token depends on all previous ones, so it is *strictly sequential*. **Memory-bound.**

</div>

Almost every serving trick — KV-cache, FlashAttention, speculative decoding — targets the slow **decode** phase.

---

# Decode re-reads the entire model, per token

To produce **one** decode token on a 70B model, the GPU must:

1. **Stream all 70B weights** from HBM — at 2 bytes (BF16) that is **140 GB**.
2. **Read the KV-cache** for the context so far (several GB).
3. **Do the math** — a few matrix–*vector* products. *Tiny* next to the data it just moved.

![w:680px](figures/lec23/svg/memory_bound.svg)

On an A100 (~1.5 TB/s HBM) moving ~150 GB takes ~0.1 s; the arithmetic finishes in a fraction of that. **The GPU sits idle, waiting for weights.** That single fact drives the whole rest of the lecture.

---

# The roofline — one picture for "memory-bound"

Plot achievable throughput against **arithmetic intensity** (FLOPs done per byte moved). Two regimes meet at a corner:

<div class="math-box">

$$\text{achievable FLOP/s} = \min\big(\underbrace{\text{peak FLOP/s}}_{\text{compute roof}},\; \underbrace{\text{bandwidth} \times \text{intensity}}_{\text{memory slope}}\big)$$

The corner (ridge) sits at intensity $=\dfrac{\text{peak FLOP/s}}{\text{bandwidth}} = \dfrac{312\text{ T}}{1.5\text{ T}} \approx \mathbf{200}$ FLOP/byte on an A100.

</div>

| Phase | Operation | Intensity | Where on the roofline |
|:--|:--|:-:|:--|
| **Prefill** | matrix–matrix | $\gg 200$ | up on the flat roof → **compute-bound** |
| **Decode** | matrix–vector | $\approx 1$ | far left on the slope → **memory-bound** |

<div class="insight">

Decode lives at the far left, using **under 1% of peak FLOPs**. You climb the roofline two ways — **move right** (bigger batches raise intensity, so each weight-read serves many sequences) or **lower the bytes** (quantization). *(Precise threshold derived in the appendix.)*

</div>

---

# Practice-check · which change speeds up decode most?

You serve Llama-70B at 8 tokens/s. Rank these by likely speedup:

<div class="popquiz">

(a) FP16 → INT4 quantization &nbsp; (b) FlashAttention-2 &nbsp; (c) speculative decoding &nbsp; (d) a faster-clocked GPU with the *same* memory bandwidth

</div>

*Answer:* roughly **a > c > b > d**. INT4 cuts the weight bytes you stream by ~4× — a direct hit on *the* bottleneck. Speculative decoding gets more tokens per weight-read. FlashAttention helps most at long context. A faster clock with the same bandwidth (d) barely moves — **because we were never compute-bound.**

---

<!-- _class: section-divider -->

## Part 3 · The KV-cache — stop re-encoding the past

---

# The KV-cache idea

<div class="insight">

**The idea.** As you generate, attention at step $t$ needs the **keys and values of every earlier token** — but $K_i, V_i$ never change once token $i$ exists. So **cache them.** Each new token computes only its *own* K/V and reuses the rest, instead of re-encoding the whole prefix every step.

</div>

![w:640px](figures/lec23/svg/kv_cache.svg)

You spend a little memory (the cache grows with context) to erase an enormous amount of repeated compute — from $O(t^2)$ down to $O(t)$ work per step.

---

# The cache grows one token at a time

![w:680px](figures/lec23/svg/kv_cache_growth.svg)

<div class="notebook">

**🎛 Interactive · watch the KV-cache grow** — step through generation token by token and see keys/values accumulate, and the compute you *skip* each step. *(Interactive Lab · `interactive/articles/kv-cache`)*

</div>

---

# MQA and GQA — shrink the cache directly

The cache size is proportional to the number of **KV heads**. So use fewer of them:

<div class="columns">
<div>

- **MHA** — every query head has its own K/V. Biggest cache.
- **MQA** — *all* query heads share **one** K/V head. Tiny cache, some quality loss.
- **GQA** — query heads share K/V in **small groups**. The sweet spot; used in Llama-2/3.

</div>
<div>

<div class="math-box">

$$M = 2 \cdot L \cdot H_{\text{kv}} \cdot d_h \cdot T \cdot B \cdot \text{bytes}$$

Only $H_{\text{kv}}$ changes: dropping from 64 heads to 8 (GQA) shrinks the cache **8×** at almost no accuracy cost.

</div>

</div>
</div>

The $Q$ side is untouched — you keep all the query heads, you just let them **share** the keys and values.

---

# KV-cache by hand · Llama 70B at 32k context

<div class="math-box">

$L = 80$ layers · $H_{\text{kv}} = 8$ (GQA) · $d_h = 128$ · $T = 32{,}000$ · $B = 1$ · 2 bytes (BF16)

$$M = 2 \cdot 80 \cdot 8 \cdot 128 \cdot 32{,}000 \cdot 1 \cdot 2 \approx \mathbf{10.5\ GB}$$

</div>

That is **for one sequence** — and it sits *on top of* the 140 GB of weights. With full MHA (64 KV heads) the same cache would be **84 GB**, blowing past any single 80 GB GPU.

<div class="insight">

The GPU runs out of **memory** long before it runs out of **FLOPs**. GQA, quantization, and paged attention all exist to move fewer of these bytes.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** A model has $L = 32$ layers, $d_h = 128$, context $T = 8192$, batch $B = 4$, BF16 (2 bytes), and uses **GQA with $H_{\text{kv}} = 4$ KV heads**.

(a) Compute the KV-cache size.
(b) What would it be with full **MHA** at $H_{\text{kv}} = 32$?
(c) What factor does GQA save here?

</div>

*Try it before the next slide.* (Use $M = 2\,L\,H_{\text{kv}}\,d_h\,T\,B\cdot\text{bytes}$.)

---

# Solution · practice problem 2

$$M = 2 \cdot 32 \cdot 4 \cdot 128 \cdot 8192 \cdot 4 \cdot 2 = 2{,}147{,}483{,}648 \text{ bytes} \approx \mathbf{2\ GB}$$

<div class="columns">
<div>

**(b) MHA, $H_{\text{kv}} = 32$**
Only $H_{\text{kv}}$ changes, $32/4 = 8\times$ larger:
$$2\text{ GB} \times 8 = \mathbf{16\ GB}$$

</div>
<div>

**(c) GQA saving**
$$\frac{32}{4} = \mathbf{8\times}$$
Same quality target, one-eighth the cache — and one-eighth the bytes streamed per token.

</div>
</div>

<div class="keypoint">

The cache scales **linearly** in every factor. Doubling context, batch, or KV heads doubles the memory — which is why long-context serving is fundamentally a memory problem.

</div>

---

# Paged attention & continuous batching — serving many users

One request rarely fills a GPU; a real server juggles **hundreds at once**. Two ideas from **vLLM** make that cheap — both are pure memory management, no change to the model:

<div class="columns">
<div>

**Paged attention.** Naive serving reserves one big *contiguous* KV buffer per request, sized for the *max* length → most of it sits empty (fragmentation). Instead, store the cache in small fixed **blocks** — like OS memory pages — with a block table mapping logical → physical. Allocate on demand, near-zero waste, and different requests can **share** blocks (a common prompt prefix is stored once).

</div>
<div>

**Continuous batching.** Don't wait for a whole batch to finish. At **every decode step** the scheduler evicts finished sequences and admits waiting ones, so the expensive weight-read always serves a full batch instead of trickling out the last straggler.

</div>
</div>

<div class="keypoint">

Together they raise GPU utilisation and crush fragmentation — vLLM serves **2–4×+** the throughput of naive `generate`. Same spine as always: waste no bytes, and make each weight-read serve as many sequences as possible (which also lifts arithmetic intensity off the roofline's floor).

</div>

---

<!-- _class: section-divider -->

## Part 4 · Quantization — how few bits can a weight survive on?

---

# The quantization ladder

<div class="insight">

**The idea.** Store each weight in **8 or 4 bits instead of 16/32** — same architecture, same trained values, just *coarser*. The model gets **2–4× smaller and faster to stream**, at a tiny, often unmeasurable accuracy cost.

</div>

![w:660px](figures/lec23/svg/quantization_ladder.svg)

Because decode is bandwidth-bound, halving the bytes per weight roughly *halves* decode latency — quantization is the single highest-leverage serving trick.

---

# INT8 — the one-number trick (a scale $s$)

<div class="insight">

**The analogy.** A tourist map can't store GPS to the millimetre. It stores **one scale** — "1 inch = 1 mile" — and every point is placed with it. Quantization does the same: one number $s$ maps a whole vector of FP32 weights onto the small integer range $[-127, 127]$.

</div>

<div class="math-box">

**1.** $s = \dfrac{\max |w|}{127}$  &nbsp;&nbsp; **2.** $w_{\text{int8}} = \text{round}(w / s)$  &nbsp;&nbsp; **3.** dequantize: $\hat w = w_{\text{int8}}\cdot s$

</div>

Store the INT8 array **plus one FP32 scale $s$ per channel**. That single shared scale is the entire cost of the compression.

---

# INT8 · worked numeric

Convert $w = [\,0.5,\; -0.2,\; 0.0,\; -1.0\,]$ from FP32 to INT8.

![w:680px](figures/lec23/svg/quantization_worked.svg)

$\max|w| = 1.0 \Rightarrow s = 1.0/127 \approx 0.007874$. Then $w/s$ rounds to $[\,64,\, -25,\, 0,\, -127\,]$.

**Stored:** 4 INT8 (4 B) $+$ one FP32 scale (4 B) $= 8$ B vs 16 B — a **2× saving** already, growing toward 4× for long vectors where the single scale is amortized. **Round-trip:** $64 \cdot s = 0.5039$ vs $0.5$ → error $\approx 0.004$, barely measurable.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Symmetric INT8-quantize $w = [\,1.2,\; -0.3,\; 0.6,\; -2.0\,]$.
(a) Find the scale $s$.
(b) Give the INT8 codes.
(c) Dequantize the **first** entry and report its round-trip error.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="columns">
<div>

**(a) Scale**
$\max|w| = 2.0$
$$s = \frac{2.0}{127} \approx 0.015748$$

**(b) Codes** ($\text{round}(w/s)$)
$1.2/s \approx 76.2 \to \mathbf{76}$
$-0.3/s \approx -19.1 \to \mathbf{-19}$
$0.6/s \approx 38.1 \to \mathbf{38}$
$-2.0/s = -127 \to \mathbf{-127}$

</div>
<div>

**(c) Round-trip, first entry**
$$\hat w_1 = 76 \times 0.015748 \approx 1.1968$$
$$|\,1.2 - 1.1968\,| \approx \mathbf{0.003}$$

<div class="insight">

The largest-magnitude weight ($-2.0$) is *exact* by construction; error only creeps into the smaller values. This is why **outliers hurt** — one huge weight stretches $s$ and coarsens everyone else.

</div>

</div>
</div>

---

# INT4 and below — protect the weights that matter

At 4 bits, naive rounding breaks. Two post-training methods keep quality:

- **GPTQ** (Frantar 2022) — quantize layer by layer, choosing codes that minimize *reconstruction* error, not raw rounding.
- **AWQ** (Lin 2023) — notice that a *tiny* fraction of weights (the ones that fire on important activations) carry most of the quality; keep **those** in higher precision.

<div class="notebook">

**🎛 Interactive · squeeze FP16 → INT8 → INT4** — slide the bit-width down and watch quality hold, then fall off a cliff. *(Interactive Lab · `interactive/articles/quantize-prune`)*

</div>

Both work well at 4-bit; AWQ edges ahead at the extreme (3-bit, 2-bit). Practical default: **AWQ 4-bit** for consumer-GPU serving.

---

# INT8 vs INT4 — the trade, worked on a 7B model

Same trained weights, coarser storage. What each bit-width actually buys:

| Precision | Bytes/wt | 7B size | Method | Quality |
|:--|:-:|:-:|:--|:--|
| FP16 | 2 | 14 GB | — | baseline |
| **INT8** | 1 | 7 GB | per-**channel** scale (LLM.int8) | ~lossless |
| **INT4** | 0.5 | 3.5 GB | **GPTQ / AWQ**, per-**group** scale | ~1–2% perplexity ↑ |

Why INT4 needs GPTQ/AWQ: 4 bits gives only **16 levels**, so a single per-channel scale is too coarse — one outlier ruins everyone else (recall practice problem 3). **GPTQ** picks codes that minimize each layer's *reconstruction* error; **AWQ** keeps the few salient weights in higher precision. Both switch to **group-wise** scales (e.g. one per 128 weights).

<div class="math-box">

**Bandwidth floor** (weights streamed at ~2 TB/s, the real decode bottleneck):
$$\text{FP16 } 14\text{ GB} \to 7\text{ ms} \quad\to\quad \text{INT8 } 7\text{ GB} \to 3.5\text{ ms} \quad\to\quad \text{INT4 } 3.5\text{ GB} \to \mathbf{1.75\ ms}$$

</div>

**4× on the bottleneck** — and $14 \to 3.5$ GB is exactly what lets a 7B model run on an **8 GB consumer GPU**. Rule of thumb: **INT8** when quality risk is unacceptable, **INT4 (AWQ/GPTQ)** when memory is the wall.

---

<!-- _class: section-divider -->

## Part 5 · FlashAttention — exact attention, no N×N matrix

---

# The attention memory problem

Naive attention *materializes* the full $N \times N$ score matrix:

```python
scores  = Q @ K.T                  # [B, H, N, N]  ← huge for long context
weights = scores.softmax(dim=-1)
out     = weights @ V              # [B, H, N, d_h]
```

For $N = 8192$ in FP16, one head's score matrix is $8192^2 \times 2 \approx 134$ MB — and it must be **written to HBM and read back** for the softmax. Across heads and layers that dominates the memory traffic. The FLOPs are fine; **the round-trips to HBM are not.**

---

# FlashAttention — tile it in SRAM

<div class="insight">

**The idea.** Compute **exact** attention without ever building the full $N\times N$ matrix — process it in small **tiles that fit in fast on-chip SRAM** (~20× faster than HBM). Same answer as naive attention, far fewer trips to slow memory.

</div>

![w:640px](figures/lec23/svg/flash_attention_tiles.svg)

<div class="insight">

**The analogy.** To paint a football-field mural you don't rent a warehouse and lay out the whole canvas — you paint one small square on a portable easel at a time, keeping just enough at the edges to line up. **You never see the whole mural at once.**

</div>

---

# How FlashAttention works (three ideas)

<div class="columns">
<div>

1. **Tile** — break $Q, K, V$ into blocks that fit in SRAM.
2. **Fuse** — do matmul + softmax + matmul for a block in *one* on-chip kernel; never write the score tile to HBM.
3. **Online softmax** — a running-max/running-sum trick updates the softmax as new blocks arrive, so the result is **exact**, not an approximation.

</div>
<div>

<div class="math-box">

**8192 context, FP16**
Naive score tile → **134 MB** in HBM.
Flash $256\times256$ tile → **131 KB** in SRAM.

*The 134 MB is never created.*

</div>

</div>
</div>

You almost never write this yourself: **PyTorch 2.0+ ships `F.scaled_dot_product_attention`**, which dispatches to a FlashAttention kernel automatically.

---

# Online softmax — why the tiles stay exact

The one trick that makes tiling *exact*. A normal softmax needs the **global** max and sum over all $N$ keys — seemingly the whole row at once. Online softmax instead keeps a **running** max $m$ and normalizer $\ell$, correcting the partial output as each block arrives:

<div class="math-box">

For a new block with local max $\tilde m$ and scores $s_j$:
$$m' = \max(m,\, \tilde m), \qquad \ell' = e^{\,m-m'}\,\ell \;+\; \sum_{j\in\text{block}} e^{\,s_j - m'}$$
and the running output is **rescaled by** $e^{\,m-m'}$ before the new block's contribution is added.

</div>

<div class="insight">

Every partial sum gets re-scaled by $e^{(\text{old max} - \text{new max})}$, so when the last block lands the result is **bit-for-bit** the full-row softmax — exact, not an approximation. Subtracting the running max also keeps it numerically stable (nothing overflows).

</div>

<div class="math-box">

**Memory:** you store only $O(d)$ running state per query, never the $O(N)$ score row. For $N=8192$ that is the **134 MB → 131 KB** difference — the big matrix is *never created*.

</div>

---

<!-- _class: section-divider -->

## Part 6 · Speculative decoding — many tokens per big-model pass

---

# Speculative decoding

<div class="insight">

**The idea.** A small, fast **draft** model guesses the next few tokens; the big model then **verifies all of them in a single forward pass**. The output is distributed *exactly* as the big model alone would produce — you just spend fewer expensive big-model steps.

</div>

![w:640px](figures/lec23/svg/speculative_decoding.svg)

The big model's one forward pass is memory-bound — you pay to stream 140 GB of weights whether you check *one* token or *five*. So checking five at once is nearly free.

---

# The concrete trace, and why it wins

**Trace** ($k = 4$): a 1B draft proposes `the cat sat on`. The 70B verifier scores all four *in one pass* and agrees with `the cat sat` — but its own 4th token is `down`, not `on`. So **accept the 3-token prefix `the cat sat`, then take the verifier's `down`** → 4 correct tokens for the cost of one big forward pass.

<div class="columns">
<div>

**Draft right** → up to $k$ tokens per big pass. **2–4× faster.**

**Draft wrong** → fall back to one token, at the cost of one cheap draft pass.

</div>
<div>

<div class="keypoint">

Correctness is preserved *exactly* by a rejection-sampling accept/reject rule — speculative decoding never changes the output distribution, only its **speed**.

</div>

</div>
</div>

---

# How much does speculative decoding actually save?

Let $\alpha$ be the probability a drafted token is **accepted**. Acceptance stops at the first miss, so with block size $k$ the expected tokens per big-model pass is

$$\mathbb{E}[\text{tokens}] = \frac{1-\alpha^{\,k+1}}{1-\alpha} \qquad (\text{derived in the appendix}).$$

<div class="columns">
<div>

**Tokens per big pass** ($k=4$):

| accept rate $\alpha$ | tokens / pass |
|:-:|:-:|
| 0.5 | 1.94 |
| 0.7 | 2.77 |
| 0.9 | 4.10 |

</div>
<div>

But it isn't free: each big pass also runs $k$ **draft** passes. With draft cost $c = \tfrac{\text{draft}}{\text{big}}$ per token,
$$\text{speedup} \approx \frac{\mathbb{E}[\text{tokens}]}{1 + k\,c}.$$
A 1B draft vs a 70B verifier gives $c\approx 0.014$, so $k=4$ adds only $\times 1.06$ overhead.

</div>
</div>

<div class="keypoint">

**Worked:** $\alpha=0.8,\, k=4 \Rightarrow \mathbb{E}\approx 3.36$ tokens/pass; after the $\times 1.06$ draft overhead, **~3.2× wall-clock**. The win is bounded by draft **quality** ($\alpha$) and draft **cheapness** ($c$) — too inaccurate *or* too big a draft erases the gain.

</div>

---

<!-- _class: section-divider -->

## Part 7 · Distillation — a train-time shrink

---

# Distillation — learn from soft labels

<div class="keypoint">

**The idea.** Train a **small student** to copy a **big teacher's full output distribution**, not just the final answer. The student stays cheap to serve while keeping most of the teacher's quality — and the shrinking happens **once, at training time**.

</div>

![w:600px](figures/lec23/svg/distillation_soft.svg)

At temperature $T=1$ the teacher is a spike — "it's a cat", nothing more. At $T=4$ it says *"cat, but dog is more cat-like than car"* — the **dark knowledge** in the wrong-class probabilities. The student matches that whole shape via $\text{KL}\big(\text{softmax}(z_t/T)\,\|\,\text{softmax}(z_s/T)\big)$. DistilBERT: 40% smaller, 60% faster, ~97% of quality.

---

# Now it clicks · the serving menu

Seven techniques, and they all answer one question — **how do I move fewer bytes per token, or avoid redoing work?**

![w:700px](figures/lec23/svg/serving_menu.svg)

<div class="keypoint">

Read the menu by its **column headers**, not the individual tricks. Every new serving optimization you meet slots into one of these buckets: *fewer bytes per weight, fewer weight reads, less recomputed attention, more tokens per pass.*

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · ES 667 efficient inference** — take a small GPT, add a KV-cache to the generation loop, then INT8-quantize the weights; measure tokens/second before and after. *(ES 667 · `notebooks/25-efficient-inference.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · distillation temperature** — soften a teacher's logits with $T$ and watch the student's target distribution spread. *(Interactive Lab · `interactive/articles/knowledge-distillation`, `softmax-temperature`)*

</div>

<div class="insight">

Real production servers (vLLM, TGI, TensorRT-LLM) stack **3–5 of these at once** — batching, paged KV-cache, quantization, FlashAttention, speculative decoding — for ~10–50× over naive PyTorch.

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Training and serving are both memory-bound — so every technique is a way to move fewer bytes, or to stop redoing work you've already done.**

</div>

- **Training fit** — mixed precision (BF16), gradient accumulation, data/tensor/pipeline parallel.
- **Serving is memory-bound** — decode re-reads *all* the weights per token.
- **KV-cache** caches past K/V; **GQA** shrinks it $\sim$8×.
- **Quantization** — one scale $s$ maps FP32 → INT8/INT4, 2–4× fewer bytes.
- **FlashAttention** — exact attention, tiled in SRAM, never builds the $N\times N$ matrix.
- **Speculative decoding** — a draft proposes, the big model verifies in one pass.
- **Distillation** — a small student inherits the teacher's soft targets.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and the standard systems literature:

- **Memory-bound framing, serving menu, KV-cache & quantization figures** — *ES 667 Deep Learning* systems materials, N. Batra · `github.com/nipunbatra/dl-teaching`
- **FlashAttention** — Dao, Fu, Ermon, Rudra, Ré, *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*, NeurIPS 2022.
- **Speculative decoding** — Leviathan, Kalman, Matias, *"Fast Inference from Transformers via Speculative Decoding"*, ICML 2023.
- **INT8 quantization** — Dettmers, Lewis, Belkada, Zettlemoyer, *"LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale"*, 2022. (INT4: GPTQ, Frantar 2022; AWQ, Lin 2023.)
- **Knowledge distillation** — Hinton, Vinyals, Dean, *"Distilling the Knowledge in a Neural Network"*, 2015.
- **Pedagogical framing** — A. Ng: *measure the bottleneck, optimize only that.*

Figures adapted from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why decode is memory-bound, precisely

Define **arithmetic intensity** $I = \dfrac{\text{FLOPs}}{\text{bytes moved}}$. A matrix–**vector** product $Wx$ with $W \in \mathbb{R}^{n\times n}$ does $\approx 2n^2$ FLOPs while reading $\approx 2n^2$ bytes (BF16 weights), so $I \approx 1$ FLOP/byte.

<div class="math-box">

A GPU is compute-bound only when $I > \dfrac{\text{peak FLOP/s}}{\text{peak bytes/s}}$. For an A100 that ratio is $\dfrac{312 \times 10^{12}}{1.5 \times 10^{12}} \approx \mathbf{200}$ FLOP/byte.

</div>

Decode's $I \approx 1 \ll 200$ — we use **under 1%** of the compute while saturating memory. The fix is not more FLOPs; it's **raising $I$** (bigger batches, so each weight-read serves many sequences) or **lowering the bytes** (quantization). Prefill, a matrix–*matrix* product, has $I \approx$ batch·seq $\gg 200$ — which is exactly why it is compute-bound.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · expected tokens per speculative pass

Let the draft propose $k$ tokens and let each be accepted independently with probability $p$. Acceptance stops at the first rejection, so the number of accepted draft tokens is a truncated geometric variable, and (adding the one "free" token the verifier always contributes) the expected tokens per big-model pass is

$$\mathbb{E}[\text{tokens}] = 1 + p + p^2 + \cdots + p^k = \frac{1 - p^{k+1}}{1 - p}.$$

<div class="math-box">

For $p = 0.7$, $k = 5$: $\;\dfrac{1 - 0.7^{6}}{1 - 0.7} = \dfrac{1 - 0.1176}{0.3} \approx \mathbf{2.94}$ tokens per pass.

</div>

So even a *mediocre* draft (accepted 70% of the time) nearly triples throughput. The real accept rule is rejection sampling, which keeps the output distribution identical to the target model's — the speedup is genuinely free of any quality cost. *(Leviathan et al. 2023, Thm 1.)*
