---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Large Language Models

## Lecture 15 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State the three **scaling laws** (compute, data, params) as power-law exponents.
2. Derive the **Chinchilla** D/N ≈ 20 optimum and explain why over-training is OK.
3. Describe **RoPE** (rotary position embedding) geometrically.
4. Explain **GQA** · fewer KV heads, same speed, same quality.
5. Contrast **DP / TP / PP** and know when to combine them.
6. Articulate **emergent abilities** · what they are and why they're contested.

---

# Where we are

- **Transformer block** (L13) · stack of attention + FFN.
- **Tokenization + pretraining** (L14) · BPE, BERT / GPT / T5.

The **Transformer** stack is the architecture. The tokenizer is the input. Now let's scale.

<div class="paper">

Today maps to the **Chinchilla paper** (Hoffmann 2022), HuggingFace course Ch 1, Karpathy's *State of GPT*. UDL Ch 12 supports this at a high level; LLM scale details come from the literature.

</div>

---

# Four questions

1. Why do LLMs keep getting **better with more compute**?
2. What changed in positional encoding — and what is **RoPE**?
3. How do 100B+ models fit on GPUs — **GQA, distributed training**?
4. What are **emergent abilities** and why are they surprising?

---

# What changed · 2018 to 2026

<div class="columns">
<div>

### Architecture

**Unchanged.** Still a decoder-only Transformer. A student from 2018 who squinted at GPT-1 would recognize Llama-3.

</div>
<div>

### Scale & engineering

Params · 117M → 1T+ (10,000×)
Data · 1B → 15T tokens (15,000×)
Compute · 10²⁰ → 10²⁵ FLOPs (100,000×)
Context · 512 → 1M tokens

</div>
</div>

<div class="keypoint">

"Bitter lesson" (Sutton) · general methods that leverage computation dominate specialized ones. LLMs are Exhibit A.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Scaling laws

The empirical backbone of the LLM era

---

# The Chinchilla result

<div class="keypoint">

Hoffmann et al. 2022 · *"Training Compute-Optimal Large Language Models"*

For a fixed compute budget C, performance is optimized when you scale **model size N** and **training tokens D** roughly **proportionally**.

$$C \approx 6 \cdot N \cdot D$$

Optimal ratio: **D / N ≈ 20 tokens per parameter**.

</div>

GPT-3 (175B params, 300B tokens · D/N ≈ 1.7) was hugely **undertrained** by this standard. Chinchilla (70B params, 1.4T tokens · D/N = 20) got better results with far fewer parameters.

---

# Chinchilla · D/N across models

![w:920px](figures/lec15/svg/chinchilla_sweet_spot.svg)

---

# Chinchilla · in one chart

![w:900px](figures/lec15/svg/chinchilla_scaling.svg)

---

# The 2023+ twist · overtraining

Modern LLMs often train *well past* Chinchilla optimal:

| Model | Params | Tokens | D/N | Notes |
|-------|--------|--------|-----|-------|
| Chinchilla | 70B | 1.4T | 20 | training-compute optimal |
| Llama 2 70B | 70B | 2T | 29 | slightly over |
| Llama 3 8B | 8B | 15T | 1875 | **wildly over** |
| Llama 3 70B | 70B | 15T | 214 | heavily over |

**Why overtrain?**

<div class="insight">

Chinchilla optimizes *training compute*. But **inference** is where models earn their keep. A smaller, over-trained model has lower inference cost per query — you get back the extra training compute many times over.

</div>

---

# Compute budget · a worked example

<div class="math-box">

Suppose you have **10²⁴ FLOPs** of training compute to spend. Chinchilla says:

- $N^* = G \cdot C^{0.5}$ · approx optimal params
- $D^* = 20 N^*$

For $C = 10^{24}$: $N^* \approx 70\text{B}$ params, $D^* \approx 1.4\text{T}$ tokens.

</div>

Change $C$ by 10× · both $N^*$ and $D^*$ grow by ~√10 ≈ 3.16×. Scale up ≠ just bigger model; bigger model + more data together.

<div class="keypoint">

This predicts · 10× more compute → ~3× bigger model trained on ~3× more tokens. Not 10× bigger. This is why GPT-4 (~1.8T) isn't 10× bigger than GPT-3 (175B).

</div>

---

# Sub-optimal training · a table

| Scenario | Params | Tokens | Status |
|:-:|:-:|:-:|:-:|
| GPT-3 (2020) · undertrained | 175B | 300B | too big, too few tokens |
| Chinchilla (2022) · optimal | 70B | 1.4T | train-compute sweet spot |
| Llama-3 8B (2024) · overtrained | 8B | 15T | inference-optimal for serving |
| A large startup's "bigger = better" model | 500B | 200B | wastes compute |

<div class="warning">

**The undertrained regime is more wasteful than the overtrained.** GPT-3 used 10× Chinchilla's compute for similar final loss. Modern LLMs carefully size $N, D$ together.

</div>

---

<!-- _class: section-divider -->

### PART 2

# RoPE · rotary positional encoding

The 2021 fix that stuck

---

# Problems with sinusoidal / learned PE

Both inject position by **adding** a position vector to the token embedding:

$$x_\text{pos}(t) = \text{emb}(token) + \text{pe}(t)$$

Problems:
- **Absolute** position only — model can't easily learn "2 positions apart" as a primitive.
- **Extrapolation** fails — trained on ≤ 4k tokens, breaks at 10k.

---

# RoPE · rotation in pictures

![w:920px](figures/lec15/svg/rope_rotation.svg)

---

# RoPE · three key properties

<div class="math-box">

1. **Relative positions encoded naturally** · inner product after rotation depends only on $m - n$ (query-minus-key position), not absolute positions.

2. **Extrapolates beyond training length** · rotation frequencies are fixed; a model trained at 4k context can extend to 32k without re-training (with minor fixes).

3. **Zero added parameters** · rotation matrices are deterministic given position; no `nn.Embedding(max_len, d_model)` allocation.

</div>

<div class="keypoint">

Llama, Mistral, Qwen, GPT-NeoX all use RoPE in 2026. A 2021 paper (Su et al.) that took ~2 years to catch on is now the default.

</div>

---

# Context length · the scaling wall

| Year | Frontier model | Context |
|:-:|:-:|:-:|
| 2018 | BERT | 512 tokens |
| 2020 | GPT-3 | 2,048 |
| 2023 | GPT-4 | 32k |
| 2023 | Claude 2 | 100k |
| 2024 | Gemini 1.5 | **1,000,000** |
| 2026 | frontier | 2-10M |

<div class="insight">

What unlocked 1M? · **RoPE extrapolation**, **FlashAttention** (O(N) memory), **GQA** (smaller KV cache), and training on long documents from the start. No single trick; the stack compounds.

</div>

---

# RoPE · rotate Q and K by position

<div class="math-box">

**Rotary Position Embedding** (Su et al. 2021)

Instead of adding a PE, **rotate** the Q and K vectors by an angle proportional to position:

$$q_m = R_{\Theta, m}\, W_q\, x_m, \quad k_n = R_{\Theta, n}\, W_k\, x_n$$

$R_{\Theta, m}$ is a block-diagonal rotation matrix. The dot product $q_m^\top k_n$ then depends only on the **relative** position $m - n$.

</div>

Why this wins:
- **Relative** positional info baked into attention directly.
- **Extrapolates** — with NTK-aware scaling, trained on 4k, works at 32k+.
- **No extra params** — RoPE is entirely deterministic.

Used in Llama 1/2/3, Mistral, Mixtral, PaLM, GPT-NeoX, most 2023+ open LLMs.

---

<!-- _class: section-divider -->

### PART 3

# Efficient attention

MQA, GQA, and the KV-cache

---

# The KV-cache problem

During autoregressive decoding, at step $t$ we need to attend to **all previous tokens** $k_{1..t-1}, v_{1..t-1}$.

We don't recompute them — we **cache** them:

$$\text{KV-cache memory} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot B \cdot \text{bytes}$$

For a 70B Llama with $L = 80, H = 64, d_h = 128$, batch $B = 1$, context $T = 32k$:

$$= 2 \cdot 80 \cdot 64 \cdot 128 \cdot 32000 \cdot 1 \cdot 2 \approx 84 \text{ GB}$$

Already more than the weights themselves. This is **the** inference-time memory pressure.

---

# MHA vs MQA vs GQA

![w:920px](figures/lec15/svg/gqa_variants.svg)

---

# Why GQA is the modern default

GQA interpolates between MHA (all heads separate) and MQA (all heads share one K/V).

- **Quality loss** from MQA: measurable; from GQA with 8 groups: ~negligible.
- **KV-cache**: 4–8× smaller than MHA.
- **Inference speed**: nearly linear in the KV-cache savings.

```python
# In Llama 2 70B:
n_heads  = 64     # query heads
n_kv     = 8      # GQA groups
d_head   = 128
# Each K and V head is shared across 8 Q heads.
```

---

<!-- _class: section-divider -->

### PART 4

# Distributed training

How you fit a 70B model on real hardware

---

# Distributed training · three parallelisms

![w:920px](figures/lec15/svg/distributed_3d.svg)

---

# Three parallelism strategies

<div class="columns">
<div>

### Data parallel (DP)

Each GPU has a **full copy** of the model, trains on different batches. Gradients averaged across GPUs.

Simple. Works for models that fit on one GPU. Breaks at 10B+.

</div>
<div>

### Tensor parallel (TP)

Split each **matrix multiply** across GPUs. Each GPU holds a slice of W.

Megatron-LM. Required for >10B. Heavy all-reduce bandwidth.

</div>
</div>

---

# Pipeline + 3D parallelism

### Pipeline parallel (PP)

Split the **layer stack** across GPUs. Layer 1-10 on GPU 1, layer 11-20 on GPU 2, etc. Bubble of idle time unless you use micro-batching.

<div class="keypoint">

**Modern training runs combine all three** (3D parallelism). Add ZeRO (sharded optimizer state) and you get the full picture.

</div>

---

# The 2026 reality

Training a 70B from scratch in 2026 · ~10k H100 GPUs for ~2 months.

- Data center cost: ~$100M
- Energy: ~1 GWh
- Engineering team: 50+

<div class="realworld">

Almost no one trains from scratch. **Everyone fine-tunes** open-weight models (Llama, Mistral, Qwen) with LoRA (next lecture).

</div>

---

<!-- _class: section-divider -->

### PART 5

# Emergent abilities

When more params unlock new behaviors

---

# Emergent abilities · the curves

![w:920px](figures/lec15/svg/emergent_abilities.svg)

---

# What "emergent" means

An ability is **emergent** if it:
1. Is near-random performance at small scale (< 10B params).
2. Rapidly improves to competent at large scale (> 50B).

No one trained specifically for it. It just appears.

---

# Emergence · the controversy

<div class="columns">
<div>

### Emergentists say

- Discontinuous jumps in capability with scale.
- New qualitative behaviors (reasoning, tool use).
- Smaller models CAN'T do these at all.

</div>
<div>

### Skeptics (Schaeffer 2023) say

- Many "emergent" curves are metric artifacts.
- Use a smoother metric (per-token log-prob vs exact match) and the curve becomes smooth.
- Still a real capability gap, but gradual.

</div>
</div>

<div class="keypoint">

Resolution · both sides are partially right. Capability improves continuously in log-probability, but certain *thresholded* tasks (match or fail) look discontinuous. The user experience is still of qualitative leaps.

</div>

---

# Chain-of-thought · prompting unlocks reasoning

<div class="math-box">

Standard prompt · "Q: 23 × 47 = ?"   →   A: "1081" (often wrong)

CoT prompt · "Q: 23 × 47 = ? Let's think step by step."   →
  A: "23 × 47 = 23 × (50 − 3) = 1150 − 69 = 1081"

</div>

CoT unlocks **multi-digit arithmetic, commonsense, logic** at 60B+. Below that, CoT adds nothing (the model can't reason in steps either).

<div class="insight">

The prompt itself is a learnable control · "let's think step by step" (Kojima 2022) can add 15 points on GSM8K. No fine-tuning. This thread becomes reasoning models (o1, Claude thinking) in 2024.

</div>

| Ability | Roughly where it emerges |
|---------|--------------------------|
| Multi-digit arithmetic | ~13B |
| Basic code generation | ~13B |
| Few-shot in-context learning | ~50B |
| Chain-of-thought reasoning | ~60B |
| Tool use (with prompting) | ~70B+ |

<div class="paper">

Wei et al. 2022 · *"Emergent Abilities of Large Language Models."* Contested (Schaeffer et al. 2023 argue it's a metric artifact) but the phenomena are real.

</div>

---

# In-context learning · the most surprising one

At pretraining, the model only learns next-token prediction. But at 100B+ params, it starts to **learn at inference time** from examples in the prompt:

```
Translate to French:
sea otter → loutre de mer
cheese → fromage
banana → banane
carrot → ???
```

The model has never seen the word "carrot" in its French dictionary. But given three examples, it figures out the task and produces "carotte".

<div class="keypoint">

This is **few-shot learning without weight updates**. Emergent at scale; the foundation of modern prompting.

</div>

---

# Chain of thought

Prompting the model to "think step by step" dramatically improves multi-step reasoning:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3
   tennis balls each. How many tennis balls?

Without CoT: "11 tennis balls." ← often wrong at small scale
With CoT:    "He starts with 5. 2 cans × 3 = 6 more.
              Total: 5 + 6 = 11." ← reliably correct
```

CoT *emerges* at scale. At 10B params, adding "let's think step by step" doesn't help. At 100B+, it adds 20+ percentage points on math benchmarks.

---

# Reasoning models (2024+)

The latest generation — **o1, o3, Claude extended thinking, DeepSeek R1** — explicitly trains the model to produce long internal chain-of-thought *before* answering:

- Trained via RL with process rewards.
- Spends 10×–100× more compute per answer.
- Often dramatically better on math, code, logic.

This is where 2026 LLMs are. We'll see alignment + RLHF in the next lecture, then peek at reasoning in L16's final slide.

---

<!-- _class: summary-slide -->

# Lecture 15 — summary

- **Chinchilla** · D/N ≈ 20 tokens per parameter is compute-optimal. Modern Llama-style models intentionally overtrain for inference gains.
- **RoPE** · rotate Q and K by position-dependent angles; relative positions baked in; extrapolates. Default in 2026 LLMs.
- **GQA** · grouped-query attention shrinks KV-cache ~4× with near-zero quality loss. Default in Llama 2 70B+.
- **Distributed training** · DP + TP + PP + ZeRO. 70B from scratch is a ~$100M engineering feat.
- **Emergence** · few-shot learning, CoT reasoning, tool use — all appear at scale, not specifically trained.

### Read before Lecture 16

HF PEFT docs; Ouyang 2022 (InstructGPT); Rafailov 2023 (DPO).

### Next lecture

**Alignment &amp; Fine-tuning** — SFT, LoRA, QLoRA, RLHF, DPO, one slide on reasoning.

<div class="notebook">

**Notebook 15** · `15-rope.ipynb` — implement RoPE from scratch; compare to sinusoidal PE on extrapolation.

</div>
