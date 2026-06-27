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
4. Explain **GQA** · fewer KV heads, near-same quality, far smaller cache.
5. Contrast **DP / TP / PP** and know when to combine them.
6. Articulate **emergent abilities** · what they are and why they're contested.

---

# Where we are

- **Transformer block** (L13) · stack of attention + FFN.
- **Tokenization + pretraining** (L14) · BPE, BERT / GPT / T5.

The **Transformer** stack is the architecture. The tokenizer is the input. Now let's scale.

<div class="paper">

**Reading & inspiration** ·
- **Karpathy · *State of GPT*** (Microsoft Build keynote) — the canonical 2023+ overview.
- **Kaplan et al. 2020 · *Scaling Laws for Neural LMs*** — the original power-law paper.
- **Hoffmann et al. 2022 · *Chinchilla*** — compute-optimal scaling.
- **Hugging Face course · Ch 1** — high-level LLM walk-through.
- **Anthropic · *Predictability of LM behavior on novel tasks***  — emergent-ability discussion.
- **Llama-3 / Mistral / Qwen technical reports** — concrete 2024-era recipes.

</div>

---

# Four questions

1. Why do LLMs keep getting **better with more compute**?
2. What changed in positional encoding — and what is **RoPE**?
3. How do 100B+ models fit on GPUs — **GQA, distributed training**?
4. What are **emergent abilities** and why are they surprising?

---

# Pop quiz · you have $1M of compute. Spend it.

You can train a Transformer LM with $C \approx 1.2 \times 10^{23}$ FLOPs. Which choice gets the best loss?

<div class="popquiz">

(a) A **70B-parameter** model on **300B tokens**.
(b) A **20B-parameter** model on **1T tokens**.
(c) A **7B-parameter** model on **3T tokens**.

Stop and pick · this is the exact question Chinchilla answered. **Answer · (b)** — at $D/N = 50$ it is the only option in the right *ballpark* of the compute-optimal rule "$D \approx 20\,N$"; (a) sits at 4.3 (way too few tokens, like GPT-3) and (c) at 429 (heavily over-trained). Most pre-2022 models were under-trained. The whole modern era is about getting *this knob* right.

</div>

---

▶ **Interactives for L15** · [softmax-temperature](https://nipunbatra.github.io/interactive-articles/softmax-temperature/) (creativity vs determinism) · [in-context-learning](https://nipunbatra.github.io/interactive-articles/in-context-learning/) (few-shot prompting live) · [mixture-of-experts](https://nipunbatra.github.io/interactive-articles/mixture-of-experts/) (Mixtral-style routing) · [mamba](https://nipunbatra.github.io/interactive-articles/mamba/) (the post-Transformer candidate).

---

# What changed · 2018 to today

<div class="columns">
<div>

### Architecture

**Unchanged.** Still a decoder-only Transformer. A student from 2018 would recognize Llama-3's *high-level* shape — though the norm, PE, and attention details were all revamped.

</div>
<div>

### Scale & engineering

As of this writing:

Params · 117M → 1T+ (10,000×)
Data · 1B → 15T tokens (15,000×)
Compute · 10²⁰ → 10²⁵ FLOPs (100,000×)
Context · 512 → 1M+ tokens

</div>
</div>

<div class="keypoint">

"Bitter lesson" (Sutton) · general methods that leverage computation dominate specialized ones. LLMs are Exhibit A.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Scaling laws

Given a fixed compute budget, how big a model should you train — and on how much data?

---

# Scaling laws · the cake-baking analogy

<div class="keypoint">

You're baking a cake (the model's performance) with a fixed budget for ingredients (compute).

Should you spend on fancy flour (more **parameters**), or on sugar/eggs (more **data**)?

Scaling laws are the **recipe** · they tell you the optimal flour-to-sugar ratio for your specific budget.

</div>

The Chinchilla paper figured out that recipe. It's: roughly **20 tokens per parameter**. Spend more on flour than that, you're "overparameterized." Spend less, "undertrained." The sweet spot maximizes the cake.

---

# The Chinchilla result

<div class="keypoint">

Hoffmann et al. 2022 · *"Training Compute-Optimal Large Language Models"*

For a fixed compute budget C, performance is optimized when you scale **model size N** and **training tokens D** roughly **proportionally**.

$$C \approx 6 \cdot N \cdot D$$

Optimal ratio: **D / N ≈ 20 tokens per parameter**.

</div>

*Fine print* · the paper actually fits $L(N, D) = E + A/N^\alpha + B/D^\beta$ and minimizes under $C = 6ND$; "20 tokens per parameter" is the headline ratio that falls out, not the whole law.

GPT-3 (175B params, 300B tokens · D/N ≈ 1.7) was hugely **undertrained** by this standard. Chinchilla (70B params, 1.4T tokens · D/N = 20) got better results with far fewer parameters.

---

# Pop quiz · how many tokens for a 1B model?

You have compute for a **1-billion-parameter** model. How many tokens should you train on?

<div class="popquiz">

(a) **1B** tokens — one token per parameter.
(b) **20B** tokens.
(c) **300B** tokens — more is always better.

Stop and compute · the answer is one multiplication away. **(b)** — apply the Chinchilla rule on the next slide, then compare with what GPT-3 actually did.

</div>

---

# Worked numeric · Chinchilla by hand

<div class="math-box">

**Rule.** $D \approx 20\,N$.

**Apply.** $N = 10^9$ → $D = 20 \times 10^9 = \mathbf{2 \times 10^{10} = 20\text{B tokens}}$.

**Budget check.** $C = 6\,N\,D = 6 \cdot 10^9 \cdot 2 \times 10^{10} = 1.2 \times 10^{20}$ FLOPs.

</div>

Now hold GPT-3 up against the same rule:

| | Params $N$ | Tokens $D$ | $D/N$ | Rule says $D$ should be |
|:-:|:-:|:-:|:-:|:-:|
| Chinchilla rule | 1B | 20B | 20 | — |
| **GPT-3 (2020)** | 175B | 300B | **≈ 1.7** | $20 \times 175\text{B} = $ **3.5T** |

<div class="insight">

GPT-3 saw 300B tokens where the rule asks for 3.5T — **~10× undertrained**. That's why Chinchilla mattered.

</div>

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

# ⭐⭐⭐ Optional · Compute budget · derivation from two rules

Two rules from Chinchilla:
1. Budget · $C = 6\,N\,D$.
2. Recipe · $D = 20\,N$.

Substitute (2) into (1) to solve for $N$ given a budget $C$:
$$C = 6\,N \cdot (20\,N) = 120\,N^2 \;\Longrightarrow\; N = \sqrt{\dfrac{C}{120}}$$

Then $D = 20\,N$.

---

# Worked example · spending $C = 1.2 \times 10^{24}$ FLOPs

1. **Solve for N.**
$N^2 = (1.2 \times 10^{24}) / 120 = 10^{22}$
$N = 10^{11}$ → **100 billion parameters**.
2. **Solve for D.**
$D = 20 \cdot 10^{11} = 2 \times 10^{12}$ → **2 trillion tokens**.

For 10× more compute · $N$ grows by $\sqrt{10} \approx 3.16$, $D$ grows by $\sqrt{10}$ too. **Both scale as $\sqrt{C}$.** This is why GPT-4 (~1.8T) isn't 10× GPT-3 (175B) — Chinchilla's recipe says spread the budget across both axes.

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

Can attention see *how far apart* two tokens are — not just where each one sits?

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

# RoPE · the spinning-pointer intuition

<div class="insight">

Imagine $Q$ and $K$ as pointers on a clock face. Instead of *adding* a position vector, **rotate** the pointer by an angle that depends on its position.
- Token at position $m = 1$ → rotate by 10°.
- Token at position $n = 3$ → rotate by 30°.

Attention score = dot product of $Q$ and $K$ → depends on the **angle between them** = $30° - 10° = 20°$, which is the relative offset $n - m = 2$.

The model learns about **relative positions directly**.

</div>

---

# ⭐⭐⭐ Optional · RoPE · derivation in 2D

For position $m$, define angle $\theta_m = m \cdot \Theta$ for some base $\Theta$. Rotation matrix:
$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

Apply to query (at position $m$) and key (at position $n$):
$q'_m = R_{\theta_m} q,\quad k'_n = R_{\theta_n} k$

Attention score:
$(q'_m)^\top k'_n = q^\top R_{\theta_m}^\top R_{\theta_n}\,k$

*Can the product $R_{\theta_m}^\top R_{\theta_n}$ be simplified? What do rotations compose to?*

---

# ⭐⭐⭐ Optional · RoPE · derivation, the punchline

Two properties of rotation matrices:
- $R_\theta^\top = R_{-\theta}$
- $R_{-\theta_m} R_{\theta_n} = R_{\theta_n - \theta_m}$

Substitute:
$(q'_m)^\top k'_n = q^\top R_{(n-m)\Theta}\,k$

**The score depends only on the relative position $n - m$, not on absolutes.**

In high-dim, group dimensions into pairs; rotate each pair with a different frequency $\Theta_i$. That's "block-diagonal."

---

# ⭐⭐⭐ Optional · Worked numeric · RoPE in 2D

$\Theta = 1$ rad. $q = [1, 2]$ at $m = 2$, $k = [3, 0]$ at $n = 3$.

**Rotation matrices.** $\theta_m = 2$, $\theta_n = 3$.
$R_2 \approx \begin{pmatrix} -0.42 & -0.91 \\ 0.91 & -0.42 \end{pmatrix},\quad R_3 \approx \begin{pmatrix} -0.99 & -0.14 \\ 0.14 & -0.99 \end{pmatrix}$

**Rotate.**
$q'_m = R_2 q \approx [-2.24,\ 0.07]^\top$
$k'_n = R_3 k \approx [-2.97,\ 0.42]^\top$

**Dot product.** $(-2.24)(-2.97) + (0.07)(0.42) \approx 6.65 + 0.03 = \mathbf{6.68}$.

*The derivation claims this should equal the relative-position form. Does it?*

---

# ⭐⭐⭐ Optional · Worked numeric · verify the relative-position claim

The relative offset is $n - m = 1$, so the derivation predicts the score equals $q^\top R_1 k$:

$$R_1 \approx \begin{pmatrix} 0.54 & -0.84 \\ 0.84 & 0.54 \end{pmatrix},\quad R_1 k = [1.62,\ 2.52]$$

$$q \cdot [1.62,\ 2.52] = 1.62 + 5.04 = \mathbf{6.66}\ \checkmark \text{ (matches 6.68 up to rounding)}$$

<div class="insight">

Two rotations, one dot product — and only the **offset** survives. That's the whole trick: relative position for free, with zero extra parameters.

</div>

---

# RoPE · three key properties

<div class="math-box">

1. **Relative positions encoded naturally** · inner product after rotation depends only on $m - n$ (query-minus-key position), not absolute positions.

2. **Extrapolates beyond training length** · rotation frequencies are fixed; a model trained at 4k context can extend to 32k without re-training (with minor fixes).

3. **Zero added parameters** · rotation matrices are deterministic given position; no `nn.Embedding(max_len, d_model)` allocation.

</div>

<div class="keypoint">

Llama, Mistral, Qwen, GPT-NeoX all use RoPE as of this writing. A 2021 paper (Su et al.) that took ~2 years to catch on is now the default.

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
| today | frontier (as of this writing) | 2-10M |

<div class="insight">

What unlocked 1M? · **RoPE extrapolation**, **FlashAttention** (O(N) memory), **GQA** (smaller KV cache), and training on long documents from the start. No single trick; the stack compounds.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Efficient attention

Why does generation run out of *memory* before it runs out of compute?

---

# KV-cache · derivation, piece by piece

During autoregressive decoding we attend to **all previous tokens**. Recomputing $K, V$ each step is wasteful → **cache** them.

Build the size, layer by layer:

1. **One token, one head, one layer** · store $K$ and $V$ vectors, each of size $d_h$.
$\text{size} = 2\,d_h$ numbers
2. **One token, one layer, all heads.** Multiply by $H$:
$\text{size} = 2\,H\,d_h$
3. **One token, all layers.** Multiply by $L$:
$\text{size} = 2\,L\,H\,d_h$
4. **All $T$ tokens.** Multiply by $T$:
$\text{size} = 2\,L\,H\,d_h\,T$ numbers
5. **In bytes.** Multiply by 2 (for fp16):
$$\text{KV-cache memory} = 2 \cdot L \cdot H \cdot d_h \cdot T \cdot B \cdot 2 \text{ bytes}$$

---

# Worked numeric · KV-cache for Llama 70B

Setup: $L = 80,\ H = 64,\ d_h = 128,\ T = 32{,}000,\ B = 1$, fp16 (2 bytes).

$$\text{Cache} = 2 \cdot 80 \cdot 64 \cdot 128 \cdot 32{,}000 \cdot 1 \cdot 2$$
$$= 32{,}000 \cdot 80 \cdot 64 \cdot 128 \cdot 4 \approx 8.4 \times 10^{10} \text{ bytes} = \mathbf{84\ \text{GB}}$$

**Punchline.** The 70B weights (fp16) take $70 \cdot 10^9 \cdot 2 = 140$ GB. The KV-cache for **one** 32k-context sequence adds another **84 GB** — more than half the model size again. This is the biggest bottleneck in long-context LLM serving. Drives GQA (next), FlashAttention (L23), and KV-cache compression.

---

# MHA vs MQA vs GQA

![w:920px](figures/lec15/svg/gqa_variants.svg)

---

# GQA · the shared-notebook analogy

<div class="insight">

The KV-cache is the model's **notebook**.
- **MHA** · 64 students each keep a private 100-page notebook → 6400 pages.
- **GQA** · 8 study groups of 8 students share one notebook each → 800 pages. Huge saving, almost no quality drop.
- **MQA** · all 64 share one notebook → 100 pages. Maximum saving, but students may overwrite each other (quality drop).

</div>

---

# How GQA shrinks the KV-cache

Refine the formula · let $H_q$ = query heads, $H_{kv}$ = key/value heads.
$$\text{Cache} = T \cdot L \cdot H_{kv} \cdot d_h \cdot 2 \cdot 2$$

Now compare for Llama 2 70B ($T = 32k$, $L = 80$, $d_h = 128$):

| Variant | $H_{kv}$ | Cache size |
|---------|----------|------------|
| **MHA** (Llama 1) | 64 | $32k \cdot 80 \cdot 64 \cdot 128 \cdot 4 \approx \mathbf{84\ \text{GB}}$ |
| **GQA** (Llama 2, 8 groups) | 8 | $32k \cdot 80 \cdot 8 \cdot 128 \cdot 4 \approx \mathbf{10.5\ \text{GB}}$ |
| **MQA** | 1 | $\approx \mathbf{1.3\ \text{GB}}$ (quality drops) |

**GQA reduces KV-cache by $H_q / H_{kv} = 64/8 = 8\times$** with negligible quality loss — the modern default (Llama 2 70B: `n_heads=64`, `n_kv=8`).

---

<!-- _class: section-divider -->

### PART 4

# Distributed training

A 70B model doesn't fit on one GPU — so how does anyone train it?

---

# Distributed training · the LEGO-team analogy

<div class="keypoint">

You need to build a massive LEGO model (the LLM) that won't fit on one person's table (one GPU). Hire a team:

- **Data Parallel** · each person builds a full copy of the model, on different parts of the data.
- **Tensor Parallel** · the whole team works on **one** giant component together (split a matrix multiply).
- **Pipeline Parallel** · set up an assembly line · person 1 does layers 1-10, person 2 does 11-20, etc.

</div>

Frontier training combines **all three** · 3D parallelism. Each axis trades off memory, compute, and communication.

---

# Distributed training · three parallelisms

![w:920px](figures/lec15/svg/distributed_3d.svg)

---

# The reality today

Training a 70B from scratch, as of this writing · ~10k H100-class GPUs for ~2 months.

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

Do bigger models just get *better* — or do they get *different*?

---

# Why "emergence" is surprising

**Null hypothesis · smooth scaling.** As you make a model bigger, training loss decreases smoothly. A 10B model is a bit better than a 1B; a 100B model is a bit better than a 10B. Intuitive.

The surprise · for some specific complex tasks, this *doesn't* happen — performance is near-random until a threshold, then takes off.

**Why?** A multi-step task is a product of step accuracies:
- Small model · 50% per step. 3-step accuracy $= 0.5^3 = 12.5\%$ · barely above random.
- Larger model · 90% per step. 3-step accuracy $= 0.9^3 = 72.9\%$ · competent!

**Smooth improvement in per-step accuracy** translates to **what looks like a discontinuous jump** in end-to-end task performance.

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

# Where abilities emerge · the scale table

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

# Chain of thought · "let's think step by step"

Prompting the model to "think step by step" dramatically improves multi-step reasoning:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3
   tennis balls each. How many tennis balls?

Without CoT: "11 tennis balls." ← often wrong at small scale
With CoT:    "He starts with 5. 2 cans × 3 = 6 more.
              Total: 5 + 6 = 11." ← reliably correct
```

CoT *emerges* at scale. At 10B params, adding "let's think step by step" doesn't help. At 100B+, it adds 20+ percentage points on math benchmarks (Kojima 2022).

<div class="insight">

The prompt itself is a control knob — no fine-tuning needed. This thread becomes **reasoning models** (next slide).

</div>

---

# Reasoning models (2024+)

The latest generation — **o1, o3, Claude extended thinking, DeepSeek R1** — explicitly trains the model to produce long internal chain-of-thought *before* answering:

- Trained via RL with process rewards.
- Spends 10×–100× more compute per answer.
- Often dramatically better on math, code, logic.

This is where frontier LLMs are as of this writing. We'll see alignment + RLHF in the next lecture, then peek at reasoning in L16's final slide.

---

<!-- _class: summary-slide -->

# Putting it all together · the L15 master sentence

<div class="math-box">

**LLMs are decoder-only Transformers from 2018, scaled.** What changed is *not* the architecture — it is **scaling laws** (loss is a power law in compute), **compute-optimal training** (Chinchilla · params ≈ tokens / 20), **modern positional encoding** (RoPE), **memory tricks** (GQA, FlashAttention), and **distributed training** (data + tensor + pipeline + ZeRO).

</div>

| Knob | What changed | Why it matters |
|:-:|:-:|:-:|
| Compute | $10^{20}$ → $10^{25}$ | scaling laws stay reliable |
| Tokens | 1B → 15T | data is the real bottleneck |
| Positional | sinusoidal → RoPE | context length scales |
| KV cache | MHA → GQA | inference memory ↓ |
| Training | single GPU → 10k+ GPUs | engineering, not architecture |

---

# Practice problems

<div class="math-box">

**P1.** Chinchilla's optimal $N \approx D / 20$. For $D = 2$ trillion tokens, what is the optimal model size? What if you have $D = 15$ trillion?

**P2.** Sketch RoPE on a 4-dim head. What does it geometrically rotate? Why is relative position the only thing the attention dot-product sees afterwards?

**P3.** **GQA** with 8 query heads sharing 2 KV heads · how many values does the KV-cache hold per token, vs vanilla MHA? Estimate the memory savings for a 70B model with context 32k.

**P4.** Sketch the **scaling-law power curve** $L(N, D, C) \approx (a/N^\alpha + b/D^\beta + c/C^\gamma)$. What does each term capture?

**P5.** Define an **emergent ability**. Pick one (e.g. multi-digit arithmetic, in-context learning, chain-of-thought) and describe how it appeared with scale.

**P6.** Why is the architecture *almost the same* across Llama-3, Mistral, Qwen, GPT-4? Is the convergence a sign of mature engineering or a missed research opportunity? Argue both sides in 4–6 lines.

</div>

---

# Lecture 15 — summary

- **Chinchilla** · D/N ≈ 20 tokens per parameter is compute-optimal. Modern Llama-style models intentionally overtrain for inference gains.
- **RoPE** · rotate Q and K by position-dependent angles; relative positions baked in; extrapolates. The current default.
- **GQA** · grouped-query attention shrinks KV-cache ~8× (64 → 8 KV heads) with near-zero quality loss. Default in Llama 2 70B+.
- **Distributed training** · DP + TP + PP + ZeRO. 70B from scratch is a ~$100M engineering feat.
- **Emergence** · few-shot learning, CoT reasoning, tool use — all appear at scale, not specifically trained.

### Read before Lecture 16

HF PEFT docs; Ouyang 2022 (InstructGPT); Rafailov 2023 (DPO).

### Next lecture

**Alignment &amp; Fine-tuning** — SFT, LoRA, QLoRA, RLHF, DPO, one slide on reasoning.

<div class="notebook">

**Notebook 15** · `15-rope.ipynb` — implement RoPE from scratch; compare to sinusoidal PE on extrapolation.

</div>

---

# The one-sentence takeaway

<div class="insight">

**Scale changed, the architecture didn't.**

*Next · a model that completes text is not yet a model that helps you — that gap is alignment.*

</div>
