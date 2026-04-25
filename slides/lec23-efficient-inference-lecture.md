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

Today maps to Chip Huyen's blog posts, HF inference docs, Dao 2022 (FlashAttention), Hinton 2015 (distillation), Leviathan 2022 (speculative decoding).

</div>

Four questions:
1. Why is LLM inference **memory-bound**, not compute-bound?
2. What is the **KV-cache** and how do we manage it?
3. What is **quantization** and how low can we go?
4. What are **FlashAttention** and **speculative decoding**?

---

<!-- _class: section-divider -->

### PART 1

# Prefill vs decode

Two phases · two bottlenecks

---

# The two phases of LLM inference

| Phase | What happens | Bottleneck |
|-------|--------------|-----------|
| **Prefill** | process the input prompt in parallel | **compute**-bound |
| **Decode** | generate tokens one at a time | **memory**-bound |

<div class="keypoint">

Prefill can use the GPU's full FLOPs — it's a big matmul. Decode is **one token per pass**, loading gigabytes of weights and KV-cache each time — mostly moving data, not computing.

</div>

Optimizing the two phases is very different. Modern inference servers (vLLM, TGI) handle them separately.

---

<!-- _class: section-divider -->

### PART 2

# The KV-cache

Where most of the memory pressure lives

---

# The KV-cache explained

![w:920px](figures/lec23/svg/kv_cache.svg)

---

# Cost with vs without cache

![w:900px](figures/lec23/svg/kv_cache_growth.svg)

---

# Why caching KV works

**Observation** · attention at step $t$ computes $Q_t K_{1:t}^\top$ · we need **every past K**, but $K_i$ doesn't change once token $i$ is generated.

<div class="keypoint">

The V vectors have the same property. So cache $K$ and $V$ as you go — each new token does **O(1) new computation for K/V** and **O(t) for the attention dot-product**, instead of O(t²) re-doing everything.

</div>

Memory grows linearly with context, but saves an order of magnitude in compute. The reason KV-cache is the first thing *any* LLM inference stack implements.

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

# Paged attention · vLLM's big idea

Kwon et al. 2023 · **vLLM** paper.

Problem · KV-cache memory is allocated contiguously; long-context requests waste space.

Solution · **paged attention** — split the KV-cache into fixed-size pages, managed like virtual memory. Each request uses only as many pages as it needs.

<div class="realworld">

vLLM and TGI both use paged attention. **~4× throughput gain** over naive implementations in production LLM serving in 2026.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Quantization

Run bigger models on smaller hardware

---

# The quantization ladder

![w:920px](figures/lec23/svg/quantization_ladder.svg)

---

# INT8 · the easy win

Converting BF16 weights to INT8 with **per-channel scaling**:

$$w_\text{int8} = \text{round}(w / s), \quad s = \max(|w_\text{channel}|) / 127$$

- Weights halved in memory.
- Matmul runs on INT8 hardware (much faster on H100, Ada).
- Quality loss · typically < 0.5% on benchmarks.

PyTorch builtin: `torch.quantization.quantize_dynamic` or `bitsandbytes` library.

---

# Worked example · quantize a single weight row

<div class="math-box">

Channel · 5 weights · `w = [0.42, -0.81, 0.05, 0.37, -0.12]`

**Step 1.** $s = \max(|w|) / 127 = 0.81 / 127 \approx 0.00638$
**Step 2.** $w_\text{int} = \text{round}(w / s) = [66, -127, 8, 58, -19]$
**Step 3.** Stored · the 5 INT8 values (5 bytes) plus one float scale $s$ (4 bytes) · 9 bytes total instead of 20 bytes for FP32.

</div>

**Reconstruction** at inference · $\hat w = w_\text{int} \cdot s$ → $[0.421, -0.810, 0.051, 0.370, -0.121]$. Max error $\sim 0.001$. Roundoff is small; quality drop on benchmarks barely measurable.

---

# INT4 and below · GPTQ / AWQ

At 4 bits per weight, naive quantization breaks. Two successful tricks:

- **GPTQ** (Frantar 2022) · post-training per-layer quantization that minimizes reconstruction loss.
- **AWQ** (Lin 2023) · protects the ~1% of weights that activate on important inputs.

Both work well at 4-bit; AWQ edges out GPTQ at extreme compression (3-bit, 2-bit).

<div class="realworld">

**2026 practical recipe** · AWQ 4-bit quantization for any LLM you're running on consumer hardware. `exllamav2` or `vLLM` both support it.

</div>

---

<!-- _class: section-divider -->

### PART 4

# FlashAttention

Rewrite attention for modern GPUs

---

# The attention memory problem

Naive attention materializes the $N \times N$ attention matrix:

```python
scores = Q @ K.T                 # [B, H, N, N]   ← huge for long context
weights = scores.softmax(dim=-1)
out = weights @ V                 # [B, H, N, d_h]
```

For $N = 8192$, a single layer needs ~8 GB just for the softmax matrix. GPU HBM bandwidth becomes the bottleneck, not FLOPs.

---

# FlashAttention · tiles in SRAM

![w:920px](figures/lec23/svg/flash_attention_tiles.svg)

---

# FlashAttention · tile and stream

<div class="paper">

Dao et al. 2022 · *"FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"*

</div>

Key ideas:

1. **Tile** · split Q, K, V into blocks that fit in SRAM (fast on-chip memory).
2. **Fuse** · compute softmax + matmul in one kernel, no intermediate materialization.
3. **Online softmax** · compute stable softmax incrementally block-by-block.

Result · **exact** attention (not approximate) with **O(N) memory** and 2–4× wall-clock speedup.

---

# FlashAttention · adoption

- PyTorch 2.0+ · built-in as `F.scaled_dot_product_attention`.
- All major inference servers use it.
- **FlashAttention 3** (2024) · further optimizations for H100 Hopper.

<div class="realworld">

Just call `F.scaled_dot_product_attention` — don't roll your own attention in 2026.

</div>

---

<!-- _class: section-divider -->

### PART 5

# Speculative decoding

Generate multiple tokens per forward pass

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

---

# Why speculative works

When the draft is right · **$k$ tokens per forward pass** instead of 1. 2–4× speedup.

When the draft is wrong · same cost as normal decoding (pay for one extra forward on the draft).

**Net effect** · draft is right ~70% of the time on typical text → ~2× speedup for free.

<div class="realworld">

Used in GPT-4, Claude 3/4, and most hosted LLMs. The "draft" is often a specially-trained small version of the verifier or a lightweight heads-only model.

</div>

---

<!-- _class: section-divider -->

### PART 6

# Knowledge distillation

Train a student to mimic a teacher

---

# Distillation · the Hinton trick (2015)

Given a big **teacher** model and a small **student** model:

<div class="math-box">

$$\mathcal{L} = \alpha \cdot \text{CE}(\text{hard labels}, \text{student}) + (1 - \alpha) \cdot T^2 \cdot \text{KL}(\text{teacher}/T, \text{student}/T)$$

- Teacher provides **soft targets** (probabilities, not one-hot).
- Temperature $T$ softens both distributions — more signal per sample.
- $T^2$ factor compensates for the softening when taking gradients.

</div>

Student learns from teacher's "dark knowledge" (relative probabilities of wrong classes), not just the right answer.

---

# Distillation in 2026

- **DistilBERT** (2019) · 40% smaller, 60% faster, 97% of BERT's quality. Still used.
- **DistilLLaMA, DistilGPT-2** · similar story for generative.
- **Modern practice** · distill a 70B teacher into a 7B student with task-specific data · near-teacher quality at 10× cheaper inference.

<div class="insight">

Many "small-but-good" 2026 models (Phi, Gemma, DistilRoBERTa) are distilled from bigger siblings. The frontier labs train big, then distill to ship.

</div>

---

<!-- _class: section-divider -->

### PART 7

# Full inference stack · 2026

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
| **Total (compounded)** | **~900×** |

</div>

<div class="keypoint">

Real production serving · ~30-100× over naive PyTorch. The rest is latency engineering (batching, KV-cache reuse across requests, model sharding).

</div>

---

# Cost economics · 2026

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

Reasoning models cost 5-10× more (inference-time compute). Same token count, much more compute per token. "Think longer" is pay-per-second.

</div>

---

<!-- _class: summary-slide -->

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
