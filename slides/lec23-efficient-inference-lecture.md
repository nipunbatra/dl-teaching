---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Efficient Inference

## Lecture 23 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain why LLM inference is **memory-bound** (not compute-bound).
2. Compute the **KV-cache size** for Llama 70B at 32k context.
3. Describe **paged attention** and why vLLM needed it.
4. Pick a **quantization level** (FP16 / INT8 / INT4 / INT2) for a deployment.
5. Explain **FlashAttention** tiling in 2 sentences.
6. Describe **speculative decoding** and its accept-rate dependency.

---

# Where we are

Training a 70B LLM costs ~$100M. But that's done *once*. **Inference** runs every request, every user, every day — and it's where models earn their keep.

<div class="paper">

**Reading & inspiration** · **Chip Huyen, *AI Engineering*** (best practical overview) · **Dao et al. 2022/2023 *FlashAttention 1–2*** · **Leviathan et al. 2022** (speculative decoding) · **Hinton et al. 2015** (distillation) + **Dettmers 2022 *LLM.int8()*** + **GPTQ/AWQ** · **vLLM blog** + **Tim Dettmers blog**.

</div>

Four questions:
1. Why is LLM inference **memory-bound**, not compute-bound?
2. What is the **KV-cache** and how do we manage it?
3. What is **quantization** and how low can we go?
4. What are **FlashAttention** and **speculative decoding**?

---

# Pop quiz · which makes inference 10× faster?

You're serving Llama-3-70B on an A100. Throughput is 8 tokens/sec. Pick **the single change** with the biggest speed-up ·

<div class="popquiz">

(a) Switch from FP16 to INT4 quantization.
(b) Use FlashAttention-2.
(c) Add speculative decoding with a 7B draft model.
(d) Move to a bigger GPU (H100).

Stop and rank. The correct rank is **a > c > b > d** for typical chat workloads · INT4 alone often gives ~$3\times$ throughput by halving memory bandwidth · speculative adds another $2\times$ · FlashAttention is best at long contexts. **Memory bandwidth is the bottleneck**, not FLOPs.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Prefill vs decode

Why does one request have *two* different bottlenecks?

---

# Reading vs writing · the inference analogy

<div class="keypoint">

Think of LLM inference like **reading a chapter** then **writing a summary**.

**Prefill** = reading the chapter all at once · all words processed in parallel.

**Decode** = writing the summary one word at a time · each new word depends on all previous, so it's strictly sequential.

</div>

That's why decode is much slower per token than prefill · we can't parallelize across the future. Almost every inference optimization (KV cache, speculative decode, FlashAttention) targets the decode phase.

---

# Prefill vs decode · the kitchen analogy

<div class="insight">

A chef preparing food.

- **Banquet (prefill)** · entire 10-course order arrives at once. Bottleneck = how fast they can chop, fry, plate **in parallel**. **Compute-bound.**
- **Single dish (decode)** · need rare ingredients in sequence — run to pantry, chop, run again. Bottleneck = the **pantry runs**, not the cooking. **Memory-bound.**

</div>

---

# Decode · why it's memory-bound

For one token of decode on a 70B model the GPU must:
1. **Fetch all 70B weights** from HBM. At 2 bytes (BF16) = **140 GB**.
2. **Fetch the KV-cache** for the context (>10 GB at 32k context).
3. **Do the math** · matrix–vector products. **Tiny** compared to the data movement.

NVIDIA A100 · ~1.5 TB/s HBM bandwidth → moving ~150 GB takes ~0.1 s. The math finishes in a fraction of that. **The GPU spends most of its time waiting for data.**

Prefill is different · the whole prompt is processed together as a big **matrix–matrix** product → the GPU's FLOPs are saturated. Compute-bound.

| Phase | Bottleneck |
|-------|-----------|
| Prefill | **compute** |
| Decode | **memory** |

Modern inference servers (vLLM, TGI) optimize the two phases separately.

---

<!-- _class: section-divider -->

### PART 2

# The KV-cache

What's actually eating all the GPU memory?

---

# The KV-cache explained

![w:920px](figures/lec23/svg/kv_cache.svg)

---

# Cost with vs without cache

![w:900px](figures/lec23/svg/kv_cache_growth.svg)

<div class="realworld">

▶ Interactive: watch the cache grow token by token — [kv-cache](https://nipunbatra.github.io/interactive-articles/kv-cache/).

</div>

---

# Why caching KV works

**Observation** · attention at step $t$ computes $Q_t K_{1:t}^\top$ · we need **every past K**, but $K_i$ doesn't change once token $i$ is generated.

<div class="keypoint">

The V vectors have the same property. So cache $K$ and $V$ as you go — each new token does **O(1) new computation for K/V** and **O(t) for the attention dot-product**, instead of O(t²) re-doing everything.

</div>

Memory grows linearly with context, but saves an order of magnitude in compute. The reason KV-cache is the first thing *any* LLM inference stack implements.

---

# Pop quiz · how big is the KV cache?

<div class="popquiz">

**Llama-7B · 4096-token context · FP16 · batch 1.** How big is the KV cache?

(a) ~20 MB &nbsp;&nbsp; (b) ~200 MB &nbsp;&nbsp; (c) ~2 GB &nbsp;&nbsp; (d) ~20 GB

**Guess before computing.** Everything you need · 32 layers · $d_\text{model} = 4096$ · standard MHA (K and V are each $d_\text{model}$ wide per layer) · 2 bytes per value in FP16.

</div>

---

# KV-cache by hand · Llama-7B

<div class="math-box">

**Per token, per layer** · $2 \;(\text{K and V}) \times 4096 \;(d_\text{model}) \times 2 \text{ bytes} = \mathbf{16\ KB}$

**Per token, all layers** · $16 \text{ KB} \times 32 \text{ layers} = \mathbf{512\ KB}$

**Full context** · $512 \text{ KB} \times 4096 \text{ tokens} = \mathbf{2\ GB}$

</div>

Answer · **(c) ~2 GB** — for **one** sequence. Half a megabyte of bookkeeping for every token generated.

---

# The punchline · memory runs out before compute

| Batch | KV cache | + weights (14 GB) | Fits on A100-40GB? |
|:-:|:-:|:-:|:-:|
| 1 | 2 GB | 16 GB | yes |
| 4 | 8 GB | 22 GB | yes |
| 16 | 32 GB | 46 GB | **no — OOM** |

<div class="insight">

The GPU runs out of **memory** long before it runs out of **FLOPs**. This is why GQA, paged attention, and quantization exist — the rest of this lecture is about moving fewer bytes.

</div>

---

# KV-cache math · Llama 70B

<div class="math-box">

$$M = 2 \cdot L \cdot H_\text{kv} \cdot d_h \cdot T \cdot B \cdot \text{bytes}$$

- $L$ · layers (80)
- $H_\text{kv}$ · KV heads (8 with GQA — see L15)
- $d_h$ · head dim (128)
- $T$ · context length (32k)
- $B$ · batch size (1)
- bytes per element (2 for BF16)

</div>

$$M = 2 \cdot 80 \cdot 8 \cdot 128 \cdot 32{,}000 \cdot 1 \cdot 2 \approx 10.5 \text{ GB}$$

With MHA (64 heads, no GQA) that would be 84 GB — **larger than the weights themselves.**

---

# Paged attention · the parking-lot analogy

<div class="keypoint">

Naive memory allocation reserves a **bus-sized parking spot** for every vehicle, even scooters · huge waste.

Paged attention has only **standard car-sized spots** ("pages"). A bus uses several; a scooter uses one. Pack many more vehicles in the same lot.

</div>

For LLMs · "vehicles" = ongoing requests, each needing variable-length KV cache. Pages let us serve many short prompts in the memory that previously held one long-prompt KV cache. ~4× throughput in vLLM.

---

# Paged attention · vLLM's big idea

Kwon et al. 2023 · **vLLM** paper.

Problem · KV-cache memory is allocated contiguously; long-context requests waste space.

Solution · **paged attention** — split the KV-cache into fixed-size pages, managed like virtual memory. Each request uses only as many pages as it needs.

<div class="realworld">

vLLM and TGI both use paged attention. **~4× throughput gain** over naive implementations in production LLM serving as of this writing.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Quantization

How few bits can a weight survive on?

---

# The quantization ladder

![w:920px](figures/lec23/svg/quantization_ladder.svg)

<div class="realworld">

▶ Interactive: squeeze FP16 → INT8 → INT4 and watch quality — [quantize-prune](https://nipunbatra.github.io/interactive-articles/quantize-prune/).

</div>

---

# INT8 · the map-scale analogy

<div class="insight">

You have a satellite image of a city with precise GPS for everything (FP32). To make a tourist map (INT8), you can't keep all that precision.

1. **Find the scale** · "1 inch on the map = 1 mile in reality." This scale $s$ is the only key piece of info.
2. **Convert** · map all real locations onto paper using $s$.

Quantization · find scale $s$ to map FP32 weights into the small range $[-127, 127]$.

</div>

---

# INT8 · derive the formula step by step

Goal · convert FP32 weights to 8-bit integers in $[-128, 127]$.

**Step 1.** Find max absolute value in the channel: $\max(|w|)$.
**Step 2.** Map the largest weight to the largest integer (127):
$\max(|w|) = s \cdot 127 \;\Rightarrow\; s = \max(|w|)/127$
**Step 3.** Quantize each weight:
$w_\text{int8} = \text{round}(w / s)$
**Step 4.** Reconstruct at inference: $\hat w = w_\text{int8} \cdot s$.

Store the int8 array + a single FP32 scale $s$ per channel.

---

# INT8 · worked numeric

Convert $w = [0.5, -0.2, 0.0, -1.0]$ from FP32 to INT8.

1. **Max abs.** $\max(|w|) = 1.0$.
2. **Scale.** $s = 1.0 / 127 \approx 0.007874$.
3. **Quantize.**
- $0.5 / s \approx 63.5 \to \mathbf{64}$
- $-0.2 / s \approx -25.4 \to \mathbf{-25}$
- $0.0 / s = 0 \to \mathbf{0}$
- $-1.0 / s = -127.0 \to \mathbf{-127}$

**Stored** · 4 INT8 (4 bytes) + scale FP32 (4 bytes) = **8 bytes** vs 16 bytes (4× FP32). **50% saving.**

**Reconstruction.** $64 \cdot 0.007874 = 0.5039$ (vs original 0.5). Max error ~0.004 — quality drop barely measurable.

PyTorch · `torch.quantization.quantize_dynamic` or `bitsandbytes`.

---

# Quantization · arithmetic in pictures

![w:920px](figures/lec23/svg/quantization_worked.svg)

---

# INT4 and below · GPTQ / AWQ

At 4 bits per weight, naive quantization breaks. Two successful tricks:

- **GPTQ** (Frantar 2022) · post-training per-layer quantization that minimizes reconstruction loss.
- **AWQ** (Lin 2023) · protects the ~1% of weights that activate on important inputs.

Both work well at 4-bit; AWQ edges out GPTQ at extreme compression (3-bit, 2-bit).

<div class="realworld">

**Practical recipe (as of this writing)** · AWQ 4-bit quantization for any LLM you're running on consumer hardware. `exllamav2` or `vLLM` both support it.

</div>

---

<!-- _class: section-divider -->

### PART 4

# FlashAttention

Can *exact* attention avoid the N×N matrix entirely?

---

# The attention memory problem

Naive attention materializes the $N \times N$ attention matrix:

```python
scores = Q @ K.T                 # [B, H, N, N]   ← huge for long context
weights = scores.softmax(dim=-1)
out = weights @ V                 # [B, H, N, d_h]
```

For $N = 8192$ in FP16 · $8192^2 \cdot 2$ bytes ≈ 134 MB *per head* — about 4 GB per layer across 32 heads. GPU HBM bandwidth becomes the bottleneck, not FLOPs.

---

# FlashAttention · tiles in SRAM

![w:920px](figures/lec23/svg/flash_attention_tiles.svg)

---

# FlashAttention · the giant-mural analogy

<div class="insight">

You need to paint a football-field-sized mural (the $N \times N$ attention matrix).

- **Naive** · rent a warehouse, lay out the canvas, paint everything. Huge temporary space (HBM).
- **FlashAttention** · paint one small square at a time, on a portable easel (SRAM). Special technique to keep edges matching. **Never see the whole mural at once.**

</div>

For $N = 32{,}768$, the attention matrix has ~1B elements → **4.3 GB per head per layer in FP32**. Too big to read/write to main memory efficiently.

---

# FlashAttention · how it works

Three ideas (Dao et al. 2022):
1. **Tile** · break $Q, K, V$ into blocks (e.g. 256×256). Each block fits in **SRAM** (~20× faster than HBM).
2. **Fuse** · do the entire (matmul + softmax + matmul) for a block in one on-chip kernel — never write the intermediate to HBM.
3. **Online softmax** · math trick to update softmax incrementally as new blocks come in. Result is **mathematically identical** to naive — not an approximation.

**Memory savings · 8192-context, FP16.**
- Naive · $8192^2 \cdot 2$ bytes ≈ **134 MB** per head/layer (must hit HBM).
- Flash · 256×256 tile · $256^2 \cdot 2 \approx \mathbf{131\ \text{KB}}$ in SRAM. The 134 MB is **never created**.

PyTorch 2.0+ ships `F.scaled_dot_product_attention` — just use it.

---

# FlashAttention · adoption

- PyTorch 2.0+ · built-in as `F.scaled_dot_product_attention`.
- All major inference servers use it.
- **FlashAttention 3** (2024) · further optimizations for H100 Hopper.

<div class="realworld">

Just call `F.scaled_dot_product_attention` — there is no reason to roll your own attention anymore.

</div>

---

<!-- _class: section-divider -->

### PART 5

# Speculative decoding

One token per forward pass — or can we do better?

---

# Speculative decoding · picture

![w:920px](figures/lec23/svg/speculative_decoding.svg)

---

# The decode speedup trick

Autoregressive generation is **one token per forward pass**. For a 70B model this caps your throughput.

**Speculative decoding** (Leviathan et al. 2022):

1. A small **draft model** (~1B params) quickly guesses the next $k$ tokens.
2. The big **verifier model** (70B) does ONE forward pass on all $k$ in parallel.
3. Accept the longest prefix where draft and verifier agree; rewind if they diverge.

**Concrete trace** ($k = 4$) · draft proposes `the cat sat on`. The 70B verifies all four in one pass and agrees with `the cat sat` but would have said `the cat sat **down**`. Accept the 3-token prefix `the cat sat`, take the verifier's `down` as the 4th — **4 tokens for the cost of one big forward pass.**

---

# Why speculative works

When the draft is right · **$k$ tokens per forward pass** instead of 1. 2–4× speedup.

When the draft is wrong · same cost as normal decoding (pay for one extra forward on the draft).

**Net effect** · draft is right ~70% of the time on typical text → ~2× speedup for free.

<div class="realworld">

Widely used in hosted LLM serving (and reportedly inside most frontier products). The "draft" is often a specially-trained small version of the verifier or a lightweight heads-only model.

</div>

---

<!-- _class: section-divider -->

### PART 6

# Knowledge distillation

Can a small student inherit a big teacher's knowledge?

---

# Distillation · the master-chef analogy

<div class="insight">

How does an apprentice learn from a master?
- **Hard labels** · master shows finished dish, says "make this." Apprentice knows only the goal.
- **Soft labels** · master says *"lots of tomato, a little basil, a hint of oregano — and crucially, more basil than oregano."* Apprentice learns relative proportions and "dark knowledge" of *what not to do*.

Distillation = method 2.

</div>

<div class="realworld">

▶ Interactive: soften the teacher's logits and watch what the student learns — [knowledge-distillation](https://nipunbatra.github.io/interactive-articles/knowledge-distillation/).

</div>

---

# ⭐⭐⭐ Optional · Distillation · the loss, term by term

Two losses:

**Part 1 · standard CE** against true labels:
$L_\text{hard} = \text{CE}(\text{labels}, \text{student})$

**Part 2 · imitate the teacher's *distribution*.** Use KL divergence between softened distributions (temperature $T > 1$):
$L_\text{soft} = \text{KL}\bigl(\text{softmax}(z_t/T)\ \|\ \text{softmax}(z_s/T)\bigr)$

Combine with weight $\alpha$:
$$\mathcal{L} = \alpha\,L_\text{hard} + (1-\alpha)\,T^2\,L_\text{soft}$$

The $T^2$ corrects for the gradient shrinkage that softening causes (so the two losses are comparable in scale).

---

# Distillation · worked numeric

Image of a cat. Classes [Cat, Dog, Car].
- Teacher logits $z_t = [10, 2, 1]$.
- Student logits $z_s = [5, 1.5, 1]$.

**$T = 1$.** softmax($z_t$) = $[0.999, 0.0003, 0.0001]$. **Almost no dark knowledge** — student barely learns relative class structure.

**$T = 4$.** Soften logits: $z_t/4 = [2.5, 0.5, 0.25]$, $z_s/4 = [1.25, 0.375, 0.25]$.
- Soft teacher · softmax($z_t/4$) ≈ $[0.88, 0.08, 0.04]$
- Soft student · softmax($z_s/4$) ≈ $[0.68, 0.17, 0.15]$

Now the student learns: increase Cat, decrease Dog and Car, **but keep Dog about 2× Car**. Rich nuanced signal that pure hard-labels would miss.

---

# Distillation today

- **DistilBERT** (2019) · 40% smaller, 60% faster, 97% of BERT's quality. Still used.
- **DistilLLaMA, DistilGPT-2** · similar story for generative.
- **Modern practice** · distill a 70B teacher into a 7B student with task-specific data · near-teacher quality at 10× cheaper inference.

<div class="insight">

Many small-but-good models as of this writing (Phi, Gemma, DistilRoBERTa) are distilled from bigger siblings. The frontier labs train big, then distill to ship.

</div>

---

<!-- _class: section-divider -->

### PART 7

# The full inference stack

How do all these tricks compose in a production server?

---

# What a production LLM server does

1. **Batched inference** · pack many user requests into each forward pass.
2. **Paged attention** for KV-cache memory efficiency.
3. **INT8/INT4 quantization** for the weights.
4. **FlashAttention** for attention compute.
5. **Speculative decoding** for throughput.
6. **Continuous batching** · new requests can join mid-batch without restart.
7. **Streaming output** · send tokens as they are generated.

Put together · ~10–50× faster and cheaper than naive implementations.

---

# The inference frameworks

| Framework | Who | Good at |
|-----------|-----|---------|
| **vLLM** | Berkeley / community | paged attention, continuous batching |
| **TGI** (Text Generation Inference) | Hugging Face | production serving |
| **llama.cpp** | community | CPU + laptop inference |
| **TensorRT-LLM** | NVIDIA | peak H100 performance |
| **MLX** | Apple | M-series Macs |
| **ExLlamaV2** | community | consumer GPU (RTX 4090) |

Choose by hardware + quality needs. For teaching, **vLLM** or **llama.cpp** are easiest to install.

---

# End-to-end · optimization stack compounded

<div class="math-box">

| Stage | Speedup vs naive |
|:-:|:-:|
| Baseline (naive fp32) | 1× |
| + bf16 | 2× |
| + KV-cache | 10× |
| + FlashAttention-2 | 3× |
| + INT8 quantization | 1.5× |
| + speculative decoding | 2.5× |
| + batching + paged attention | 4× |
| **Naive product (over-counts)** | ~900× |

</div>

<div class="keypoint">

Those factors **don't multiply cleanly** — they overlap and compete for the same bottleneck. Real production serving lands at **~30–100× over naive PyTorch**. The rest is latency engineering (batching, KV-cache reuse across requests, model sharding).

</div>

---

# Cost economics · a snapshot (as of this writing)

<div class="math-box">

| Model | Cost per 1M input tokens | Cost per 1M output tokens |
|:-:|:-:|:-:|
| Claude 3.5 Haiku | $1 | $5 |
| GPT-4o-mini | $0.15 | $0.60 |
| Llama-3 70B (self-hosted) | ~$0.30 | ~$0.60 |
| Claude 3.5 Sonnet | $3 | $15 |
| o1 | $15 | $60 |

</div>

<div class="realworld">

Prices move fast — treat these as orders of magnitude, not quotes. Reasoning models cost 5-10× more (inference-time compute) · "think longer" is pay-per-second.

</div>

---

<!-- _class: summary-slide -->

# Putting it all together · the L23 master sentence

<div class="math-box">

**LLM inference is memory-bound, not compute-bound** · the GPU sits idle waiting for weights to stream from VRAM. Every speed-up is a way to **move fewer bytes** · **quantization** (less per-weight) · **FlashAttention** (reuse KV in SRAM) · **PagedAttention/vLLM** (batch concurrent users) · **speculative decoding** (verify many tokens in one forward pass) · **distillation** (smaller student model). **Compress what you load · share what you compute.**

</div>

This is the production toolkit as of this writing — every commercial LLM endpoint uses 3–5 of these.

---

# The toolkit · saves vs cost

| Trick | Saves | Cost |
|:-:|:-:|:-:|
| INT4 quantization | $4\times$ memory bandwidth | small accuracy drop |
| FlashAttention-2 | reads/writes for attn | none |
| PagedAttention (vLLM) | KV-cache fragmentation | engineering complexity |
| Speculative decoding | latency on chat | extra draft model |
| Distillation | size + cost | training run |

---

# Practice problems · 1

<div class="math-box">

**P1.** Compute the **arithmetic intensity** (FLOPs / bytes) of a single decoding step on a 70B model. Show it is well below the GPU's roofline ratio · explain why this means *memory-bound*.

**P2.** **KV-cache** for a 70B model with 80 layers, $d_\text{model}=8192$, GQA(8), context $32{,}768$, FP16. Compute the cache size in GB. Compare to model weights.

**P3.** **FlashAttention** keeps Q, K, V tiles in SRAM and recomputes softmax in chunks. Why does this avoid materializing the $O(N^2)$ attention matrix?

</div>

---

# Practice problems · 2

<div class="math-box">

**P4.** **Distillation** trains a small student to match a large teacher's softmax with temperature $T$. Why is $T > 1$ used during training? What does that do to the loss surface?

**P5.** **INT4 weight quantization with FP16 activations** · sketch the dequant-on-the-fly inner loop. Why don't we quantize activations to INT4 too?

**P6.** ⭐⭐⭐ *(stretch)* Speculative decoding · draft proposes $k = 5$ tokens · target verifies in one pass · accepts the longest matching prefix. If acceptance rate is $p$, expected tokens per pass is $1 + p + p^2 + \ldots + p^k$. Compute for $p = 0.7$, $k = 5$.

</div>

---

# Lecture 23 — summary

- **Prefill vs decode** · compute-bound vs memory-bound. Different optimizations.
- **KV-cache** · dominates memory for long contexts. GQA (L15) + paged attention shrink it.
- **Quantization** · BF16 → INT8 → NF4. AWQ handles 4-bit well; <1% quality loss.
- **FlashAttention** · exact attention with O(N) memory · 2–4× faster.
- **Speculative decoding** · draft model proposes, big model verifies in parallel · 2–4× throughput.
- **Distillation** · big teacher trains small student on soft targets.
- **Production stack** · all of these combined → ~10–50× over naive.

### Read before Lecture 24

Anthropic interp blog; Chi et al. 2023 (Diffusion Policy); blog posts on Claude Code / computer use.

### Next lecture · last one!

**Frontier · Agents, Reasoning, Interpretability + course wrap-up.**

<div class="notebook">

**Notebook 23** · `23-kv-cache.ipynb` — take a small GPT; add KV-cache to generation loop; measure tokens/second speedup.

</div>

---

# The one-sentence takeaway

<div class="insight">

**Inference is memory-bound, not compute-bound.**

*Next lecture — the last one: agents, reasoning models, interpretability, and what nobody has solved yet.*

</div>
