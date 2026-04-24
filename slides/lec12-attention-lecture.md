---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# The Attention Mechanism

## Lecture 12 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain attention as **differentiable dictionary lookup**.
2. Derive **scaled dot-product** and justify the √d_k.
3. Distinguish **Bahdanau (additive)** from **Luong (multiplicative)**.
4. Implement **QKV self-attention** in PyTorch.
5. Apply **causal masking** to get a GPT-style decoder.
6. State the **O(n²) complexity** wall and its consequences.

---

# Recap · where we are

Module 6 opens. The previous lecture ended on a cliff-hanger:

- **Seq2Seq** works for short sentences.
- The **fixed-size context vector** can't hold everything for long ones.
- **BLEU collapses** past ~30 tokens.

Today we fix it.

<div class="paper">

Today maps to **Prince Ch 12** (early sections). This is the single most influential idea in deep learning between backprop and diffusion.

</div>

---

# The one-line fix

<div class="keypoint">

Don't force the decoder to read from one fixed vector. Let it **peek at every encoder state** and decide which ones matter for the current step.

</div>

Bahdanau et al. 2014 · "Neural Machine Translation by Jointly Learning to Align and Translate."

---

# Why one vector can't hold everything

Think about translating a 40-word English sentence into French. A Seq2Seq encoder has to compress:

- every word's identity
- the full syntactic structure
- the tense, number, gender, mood of every verb and noun
- the coreference chains ("it" refers to "the box")
- the sentiment

...into a **single 512-dim vector**. One number for every 0.3 bits of meaning.

<div class="warning">

No matter how big you make that vector, there's always a sentence that exceeds it. The information-theoretic problem is **fundamental**, not an engineering issue.

</div>

---

# The shift in viewpoint

<div class="columns">
<div>

### Old · push mode

Encoder *pushes* a summary forward. Decoder takes whatever fits.

- One-shot summarization.
- Lossy — compression is mandatory.

</div>
<div>

### New · pull mode

Decoder *pulls* information on demand. Encoder keeps everything around.

- No compression required.
- Decoder chooses what's relevant *per step*.

</div>
</div>

Attention is the mechanism for the *pull* — a differentiable version of "look up the word I need right now."

---

# Four questions

1. What does attention look like — literally, as a heatmap?
2. What are Q, K, V and why "retrieval"?
3. Why do we divide by $\sqrt{d_k}$?
4. What is **self**-attention and how does it differ from cross-attention?

---

<!-- _class: section-divider -->

### PART 1

# Attention as soft alignment

The heatmap view

---

# What attention looks like

![w:720px](figures/lec12/svg/attention_heatmap.svg)

---

# Interpretation

Every target word computes a **distribution over source words**. Darker cell = more attention mass.

Three things to notice:

1. **Rough diagonal** — most alignments are monotonic.
2. **"traversé" attends to "cross"** — semantic alignment across a language barrier.
3. **"qu'il" attends to BOTH "it" AND "animal"** — coreference resolution, emergent from training.

<div class="insight">

No one told the model what "it" refers to. It learned this by minimizing translation loss. Attention made linguistic structure visible for the first time.

</div>

<div class="realworld">

▶ Interactive: hover over source tokens, see attention weights in real time — [attention](https://nipunbatra.github.io/interactive-articles/attention/).

</div>

---

<!-- _class: section-divider -->

### PART 2

# From Bahdanau to QKV

Two parameterizations · one abstraction

---

# A 3×3 worked example · before the math

Source · "the cat slept"      Target step · decoding French for "cat" → *"chat"*

<div class="math-box">

Encoder states (toy 2-d): $h_1 = [1, 0]$ (the), $h_2 = [0, 1]$ (cat), $h_3 = [0.2, 0.1]$ (slept)

Decoder state: $s_t = [0.1, 0.9]$ (about to emit *chat*)

Raw scores (dot products): $s_t \cdot h_1 = 0.1$, $s_t \cdot h_2 = 0.9$, $s_t \cdot h_3 = 0.11$

Softmax: $[0.24, 0.54, 0.22]$ — *"cat"* gets the largest weight.

Context: $c_t = 0.24 h_1 + 0.54 h_2 + 0.22 h_3 = [0.29, 0.56]$

</div>

The decoder reads a **weighted mixture**, dominated by the relevant word. That's one step of attention.

---

# Bahdanau (additive) attention · 2014

Compute an alignment score between decoder state $s_{t}$ and encoder state $h_i$ via a **learned network**:

<div class="math-box">

$$e_{t,i} = \mathbf{v}^\top \tanh(W_1 h_i + W_2 s_t)$$

$$\alpha_{t,i} = \text{softmax}_i(e_{t,i})$$

$$c_t = \sum_i \alpha_{t,i}\, h_i$$

</div>

- $e_{t,i}$ · how well does encoder state $i$ match decoder state $t$?
- $\alpha_{t,i}$ · normalized weights (sum to 1 over $i$).
- $c_t$ · new context — a weighted combo of encoder states, recomputed every decoder step.

---

# Luong (multiplicative) attention · 2015

A simpler score, no learned MLP:

$$e_{t,i} = s_t^\top h_i$$

- **Faster** — one matrix multiply instead of a two-layer MLP.
- **Equivalent in expressiveness** when you have enough data (the projections in QKV absorb Bahdanau's $W_1, W_2$).

This is the version Vaswani et al. kept for the Transformer in 2017, with one small but crucial addition: the $\sqrt{d_k}$ scaling.

---

# Why multiplicative won

Two reasons the ML community moved from Bahdanau to Luong:

1. **Hardware** — a dot product is one matrix multiply; GPUs love it. Bahdanau's MLP has element-wise tanh, which is slower per op and harder to batch.
2. **Enough capacity elsewhere** — once we added learned $W_Q, W_K$ projections (next section), we no longer needed the $\tanh(W_1 h + W_2 s)$ to *learn* the similarity. The dot product of projections already gives it.

<div class="insight">

Pattern in DL · keep the core operation small + fast, push learning into the linear layers around it. This is also why attention beat CNN for sequences — CNN's inductive bias was baked in; attention's bias is *learned*.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Q, K, V · the clean abstraction

Attention as database retrieval

---

# Soft retrieval · picture

![w:920px](figures/lec12/svg/attention_soft_retrieval.svg)

---

# The retrieval metaphor

Imagine a Python dictionary lookup:

```python
db = {"cat": "meow", "dog": "bark"}
query = "cat"
result = db[query]      # returns "meow"
```

Attention is the **soft** version of this:

- **Query** — what you're looking for (from the decoder).
- **Key** — what each encoder state announces itself as.
- **Value** — what each encoder state actually *contains*.

Score keys against the query → softmax → use weights to blend values.

---

# Soft retrieval · why three roles not one

You could imagine an attention mechanism where $K$ and $V$ are the same thing. Early models did exactly this (Luong 2015). So why separate them?

<div class="keypoint">

**Keys say "what I am"; values say "what I contribute."**

A word like "bank" in a sentence should be *found* by the query "financial", but *contribute* the full contextual embedding. Keys for retrieval, values for content.

</div>

- $K$ optimized for similarity with plausible queries.
- $V$ optimized to carry whatever downstream layers need.

Separating them doubles the parameter count of one head but roughly doubles the expressiveness too.

---

# The Python-dict mental model · extended

```python
# Hard retrieval
db = {k1: v1, k2: v2, k3: v3}
out = db[query]                      # exact match → one value

# Soft retrieval (attention)
scores  = [sim(query, k) for k in [k1, k2, k3]]   # similarities
weights = softmax(scores)                          # probabilities
out     = weights[0]*v1 + weights[1]*v2 + weights[2]*v3
```

<div class="insight">

Attention is **differentiable dictionary lookup**. The network's parameters shape what "sim" means and what each key/value represents. Everything else is the soft version of `db[query]`.

</div>

---

# QKV · the computation

![w:920px](figures/lec12/svg/qkv_attention.svg)

---

# Scaled dot-product · step by step

![w:920px](figures/lec12/svg/attention_scaled_dotproduct.svg)

---

<!-- _class: code-heavy -->

# QKV · projections from the same input

Crucially, $Q, K, V$ are all **linear projections of the input**:

<div class="math-box">

$$Q = X\, W_Q, \quad K = X\, W_K, \quad V = X\, W_V$$

$W_Q, W_K, W_V$ are learned $d \times d_k$ matrices.

</div>

The network *learns* what each role should be. The same input plays three parts depending on which projection you apply.

```python
class AttentionHead(nn.Module):
    def __init__(self, d_in, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_k, bias=False)
        self.Wk = nn.Linear(d_in, d_k, bias=False)
        self.Wv = nn.Linear(d_in, d_k, bias=False)

    def forward(self, x):
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        weights = scores.softmax(dim=-1)
        return weights @ V
```

---

<!-- _class: section-divider -->

### PART 4

# Why $\sqrt{d_k}$?

The scaling that makes attention work

---

# The problem with unscaled dot products

For $Q_i, K_j \sim \mathcal{N}(0, 1)$, independent, in $d_k$ dimensions:

$$Q_i^\top K_j = \sum_{k=1}^{d_k} Q_{i,k} K_{j,k}$$

**Variance of the sum** = $d_k$. So dot products scale as $\sqrt{d_k}$.

With $d_k = 512$: raw scores are $\sim \pm 22$. Softmax of $[22, -22, 22, \ldots]$ → nearly one-hot.

---

# Numeric demo · softmax at different scales

<div class="math-box">

Raw logits $[s_1, s_2, s_3] = [2.0, 1.0, 0.5]$. Softmax behaves nicely:

| temperature | softmax |
|:-:|:-:|
| /1    | (0.58, 0.21, 0.13, ...) — soft |
| /4    | (0.40, 0.30, 0.26, ...) — very soft |
| ×10   | (0.9999, 4e-5, 2e-7) — one-hot |

</div>

Dot products without scaling behave like the bottom row — **effectively one-hot**. Divide by $\sqrt{d_k}$ and you land back on the top row. The scaling is doing exactly the role of a temperature denominator, derived from variance analysis rather than tuned by hand.

---

# In pictures

![w:920px](figures/lec12/svg/sqrt_dk_scaling.svg)

---

# Why one-hot is bad

If softmax outputs are (nearly) one-hot, then attention picks **one** encoder state and ignores the rest.

**Two consequences:**
1. Information from other positions is discarded.
2. Gradient through the softmax is near zero (saturation) → training stalls.

**The fix** — divide scores by $\sqrt{d_k}$:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Scores stay in a healthy range → softmax stays soft → gradients keep flowing.

---

# The $\sqrt{d_k}$ derivation in three lines

Assume $Q, K$ entries are i.i.d. with zero mean and unit variance.

$$\mathbb{E}[Q^\top K] = 0, \quad \text{Var}[Q^\top K] = \sum_{k=1}^{d_k} \text{Var}[Q_k K_k] = d_k$$

So $Q^\top K \sim \mathcal{N}(0, d_k)$. Dividing by $\sqrt{d_k}$ makes the variance **1** — independent of dimension.

<div class="keypoint">

**Why this matters** · the same attention block can be used at $d_k = 64$ or $d_k = 4096$ without retuning temperatures. The scaling is *dimension-invariant* by construction.

</div>

---

<!-- _class: section-divider -->

### PART 5

# Self-attention vs cross-attention

Same machinery · different sources for QKV

---

# Cross-attention · the decoder reads the encoder

In a Seq2Seq + attention model:

- Queries come from the **decoder** (current target position).
- Keys and values come from the **encoder** (all source positions).

This is what the first attention heatmap showed — one distribution per target step over source positions.

---

# Self-attention · a sequence attends to itself

Same operation. Now:

$$Q = X\, W_Q, \quad K = X\, W_K, \quad V = X\, W_V$$

All three come from **the same input sequence**.

<div class="keypoint">

Self-attention lets every position in a sequence aggregate information from every other position — **in parallel**, with no recurrence.

</div>

This is the idea that killed RNNs and gave us Transformers (L13).

---

# Self-attention · 3 tokens, by hand

Sentence · "the cat slept"     token embeddings $x_1, x_2, x_3 \in \mathbb{R}^d$.

After projecting: $Q, K, V$ are each a $3 \times d_k$ matrix.

<div class="math-box">

$$QK^\top / \sqrt{d_k} = \begin{bmatrix} s_{11} & s_{12} & s_{13} \\ s_{21} & s_{22} & s_{23} \\ s_{31} & s_{32} & s_{33} \end{bmatrix}$$

Row $i$ says: *how much does token $i$ want to look at every other token?*

Softmax per row → 3×3 attention matrix $A$. Output $A V$ is again $3 \times d_k$.

</div>

Every output row is a weighted blend of value rows — a contextualized embedding for that token.

---

# Self-attention vs convolution · same goal, different bias

<div class="columns">
<div>

### Convolution

- Each output depends on a **fixed** local window.
- Inductive bias: *locality*.
- Parameters: shared kernel.

</div>
<div>

### Self-attention

- Each output depends on **all** positions.
- Inductive bias: *learned* from data.
- Parameters: Q, K, V projections.

</div>
</div>

<div class="insight">

Convolution bakes in "nearby tokens matter"; self-attention lets the network decide from data whether nearby or far-away tokens matter. When you have lots of data, *learned* bias wins over *hand-designed* bias. That's the whole arc of the 2017–2025 vision revolution in one sentence.

</div>

---

# Causal self-attention · two lines to make GPT

To make attention *autoregressive* (can't peek at future tokens), mask out the upper triangle before softmax:

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)     # (n, n)
mask   = torch.triu(torch.ones_like(scores), diagonal=1).bool()
scores.masked_fill_(mask, float('-inf'))              # future → -inf
weights = scores.softmax(dim=-1)                      # rows still sum to 1
```

<div class="keypoint">

**That's the only difference between BERT-style (bidirectional) and GPT-style (causal) attention.** Same module, different mask. L13 uses this trick to build a decoder.

</div>

---

# Complexity · the O(n²) wall

Self-attention on a sequence of length $n$:
- $Q K^\top$ builds an $n \times n$ matrix → **$O(n^2)$ memory** and **$O(n^2 d)$ compute**.
- Double the context → 4× the cost.

<div class="warning">

At $n = 8{,}192$ and $d = 4096$, one head's attention matrix is already **64 MB per layer**. Scaling context to 1M tokens naively would need 500 GB per layer. This is the wall that motivates:

- **FlashAttention** (L23) · recompute attention in tiles, avoiding the full $n \times n$ matrix.
- **Sparse / local / linear attention** (reading) · trade off quality for $O(n \log n)$ or $O(n)$.
- **KV caching** (L23) · don't redo the whole computation at every generation step.

</div>

---

# The four kinds of attention you will meet

| Type | Q comes from | K, V come from | Used in |
|------|--------------|-----------------|---------|
| Encoder self-attn | encoder | encoder | Transformer encoder |
| Decoder self-attn (causal) | decoder (past only) | decoder (past only) | GPT, decoder of Transformer |
| Cross-attention | decoder | encoder | Seq2Seq decoder, translation |
| Masked/local attention | input | input (masked) | Longformer, Reformer, etc. |

---

<!-- _class: section-divider -->

### PART 6

# What attention unlocked

One slide of consequences

---

# Why attention was such a big deal

1. **Bottleneck solved** · no more "fit everything into 512 dims."
2. **Long-range dependencies** · every target step can see any source step.
3. **Parallelizable** · unlike RNNs, all attention scores can be computed at once.
4. **Interpretable** · attention heatmaps are the first DL visualization tool that's actually informative.
5. **Transfer** · attention blocks compose cleanly — stack them, mix them with cross-attention, make them multi-headed. The Transformer (L13) is exactly this.

<div class="realworld">

Without attention · no Transformer · no BERT · no GPT · no Claude · no diffusion text conditioning. A single 2014 paper seeded the next decade.

</div>

---

# "Attention Is All You Need" · the 2017 pivot

Vaswani et al.'s one-page insight · drop the RNN entirely, use **only** attention plus FFNs plus positional encodings.

<div class="paper">

Before: encoders and decoders were RNNs *with* attention as a helper. After: attention was the load-bearing operation; RNN was gone.

</div>

Consequence table:

| Axis | RNN+attention | Transformer |
|------|---------------|-------------|
| Sequential compute | yes (unrolled) | no (parallel) |
| Long-range path | $O(n)$ hops | $O(1)$ hops |
| Training throughput | slow | 10–20× faster |
| Scaling | plateaus at ~1B | trained to 1T+ |

Every major model since 2018 (BERT, GPT-*, T5, Claude, Llama) is this architecture, plus or minus details. **L13 builds it from parts.**

---

<!-- _class: summary-slide -->

# Lecture 12 — summary

- **Attention** = soft retrieval · each query selects a weighted combination of values based on similarity to keys.
- **Bahdanau** (additive, 2014) and **Luong** (multiplicative, 2015) are two parameterizations — we use Luong's dot-product form in Transformers.
- **QKV abstraction** · $Q, K, V$ are learned projections of the same input; the network decides what each role should be.
- **$\sqrt{d_k}$ scaling** — without it, softmax collapses to one-hot at large $d_k$ and gradients die.
- **Self-attention** · Q, K, V from the same sequence; parallel, long-range, interpretable.
- **This unlocked the Transformer** — next lecture.

### Read before Lecture 13

**Prince Ch 12** mid-sections (Transformer block).

### Next lecture

**The Transformer — built live.** Multi-head attention · positional encoding · residual + LayerNorm · the full encoder-decoder stack.

<div class="notebook">

**Notebook 12** · `12-attention-nmt.ipynb` — add attention to Lecture 11's Seq2Seq; visualize attention heatmaps; watch BLEU improve on long sentences.

</div>
