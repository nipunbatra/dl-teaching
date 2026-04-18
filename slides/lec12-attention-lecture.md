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

<!-- _class: section-divider -->

### PART 3

# Q, K, V · the clean abstraction

Attention as database retrieval

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

# QKV · the computation

![w:920px](figures/lec12/svg/qkv_attention.svg)

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
