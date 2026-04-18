---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# The Transformer — Built Live

## Lecture 13 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Recap · where we are

Last lecture: **attention** fixed Seq2Seq's bottleneck. Q, K, V scaled dot product, softmax, weighted values.

**Today we stack it all together.**

<div class="paper">

Today maps to **Prince Ch 12** (mid sections). Backup video: Karpathy's *Let's build GPT: from scratch, in code, spelled out* (YouTube).

</div>

---

# Four questions

1. What's in a full Transformer block — and why that order?
2. Why **multi-head** attention instead of one big head?
3. How does the model know position if there's no recurrence?
4. What's the difference between encoder, decoder, and decoder-only models (GPT)?

---

<!-- _class: section-divider -->

### PART 1

# The block

Two sublayers, two residuals, two norms

---

# The Transformer block (pre-norm)

![w:720px](figures/lec13/svg/transformer_block.svg)

---

# Why this exact structure?

<div class="keypoint">

**Every modern Transformer is this block, stacked 10–100 times.** BERT, GPT, Llama, Claude — same structure, different depths, widths, and data.

</div>

Three ingredients you've already seen in isolation:

- **Attention** (L12) — sequence-to-sequence mixing.
- **Residual connections** (L2) — gradient highway.
- **LayerNorm** (L6) — scale drift control.

The genius was **gluing them together** into one block you can safely stack.

---

<!-- _class: code-heavy -->

# The block in PyTorch · 20 lines

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        # Pre-norm, then attention, then residual
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + a

        # Pre-norm, then FFN, then residual
        x = x + self.ffn(self.norm2(x))
        return x
```

That's the Transformer. Everything else is plumbing around this.

---

<!-- _class: section-divider -->

### PART 2

# Multi-head attention

Why one head is never enough

---

# The multi-head idea

![w:920px](figures/lec13/svg/multi_head_attention.svg)

---

# Why split into heads?

A single attention head has to choose *one* distribution over positions per query. But real language has **multiple relations** to track at once:

- Syntax — who modifies whom
- Semantics — word meaning similarity
- Coreference — pronoun resolution
- Position — relative distance

<div class="keypoint">

Multiple heads = multiple "attention circuits" running in parallel. Each head specializes in a different kind of relationship.

</div>

Empirically, 8 or 16 heads is standard. Increasing beyond has diminishing returns — each head's dim $d_k = d / h$ gets too small to be useful.

---

# Multi-head attention in PyTorch

```python
# PyTorch gives you this in one line:
self.attn = nn.MultiheadAttention(d_model=512, num_heads=8, batch_first=True)

# By hand, the core operation is:
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads):
    B, N, d = x.shape
    d_k = d // n_heads

    # Project and reshape: (B, N, d) → (B, n_heads, N, d_k)
    q = (x @ Wq).view(B, N, n_heads, d_k).transpose(1, 2)
    k = (x @ Wk).view(B, N, n_heads, d_k).transpose(1, 2)
    v = (x @ Wv).view(B, N, n_heads, d_k).transpose(1, 2)

    # Scaled dot-product attention per head, then concatenate
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    weights = scores.softmax(dim=-1)
    out = (weights @ v).transpose(1, 2).contiguous().view(B, N, d)

    return out @ Wo
```

---

<!-- _class: section-divider -->

### PART 3

# Positional encoding

Telling the model "this is position 7"

---

# The problem · attention is permutation-invariant

Attention with no positional info:

> "Dog bites man"   →   same attention weights as
> "Man bites dog"

Both have the same tokens, just reordered. Attention computes $QK^\top$ — no notion of *where* each token sits.

<div class="keypoint">

We need to inject **position information** into the token embeddings. The Transformer's choice was sinusoidal — a specific multi-scale "clock" vector added to each token's embedding.

</div>

---

# Sinusoidal positional encoding

![w:900px](figures/lec13/svg/positional_encoding.svg)

---

# Two modern alternatives

| Method | How | Used in |
|--------|-----|---------|
| **Sinusoidal** (Vaswani 2017) | fixed sin/cos | original Transformer, BERT |
| **Learned** | `nn.Embedding(max_len, d_model)` | GPT-2, GPT-3 |
| **RoPE** (Su 2021) | rotate Q and K by position-dependent angle | Llama, Mistral, PaLM, modern LLMs |
| **ALiBi** (Press 2021) | bias attention scores by relative distance | OPT-175B, some variants |

<div class="realworld">

2026 · **RoPE** dominates new LLMs. We'll cover it in L15 (LLMs). For now, any of the four works — pick the one that matches your base model.

</div>

---

<!-- _class: section-divider -->

### PART 4

# The full architecture

Encoder · decoder · causal mask

---

# Encoder-decoder (original Vaswani 2017)

![w:820px](figures/lec13/svg/full_transformer.svg)

---

# Three architectural flavours

| Model | What it is | Use case |
|-------|------------|----------|
| **Encoder-only** (BERT) | stack of encoder blocks | classification, embedding, retrieval |
| **Decoder-only** (GPT, Llama, Claude) | stack of decoder blocks, **no cross-attn** | autoregressive generation |
| **Encoder-decoder** (T5, BART) | both, with cross-attention | translation, summarization |

In 2026, **decoder-only** dominates LLMs. Encoder-only ships in retrieval pipelines. Encoder-decoder survives for translation-style tasks.

---

# Causal (autoregressive) masking

For a **decoder** generating text one token at a time, we need to prevent each position from attending to *future* tokens during training.

<div class="math-box">

**Causal mask** — add $-\infty$ to upper triangle of attention scores *before* softmax:

$$\text{scores}_{i,j} = \begin{cases} QK^\top_{i,j}/\sqrt{d_k} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$

After softmax, the $-\infty$ positions become 0 — token $i$ ignores everything after it.

</div>

One triangle mask, one line of code. This is what makes GPT autoregressive.

---

# Masking in PyTorch

```python
# Causal mask for sequence length N
N = 128
mask = torch.triu(torch.ones(N, N), diagonal=1).bool()   # upper-triangular, excluding diagonal
#  [[F, T, T, T, ...],
#   [F, F, T, T, ...],
#   [F, F, F, T, ...],
#   ...]
# True = mask (set score to -inf)

# In MultiheadAttention, mask=True means "block this position"
out, _ = self.attn(x, x, x, attn_mask=mask)
```

---

<!-- _class: section-divider -->

### PART 5

# Put it all together · build GPT-tiny

Karpathy nanoGPT in 80 lines

---

# nanoGPT · 80 lines that changed the world

```python
class GPT(nn.Module):
    def __init__(self, vocab, d_model=192, n_heads=6, n_layers=6, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks  = nn.ModuleList([TransformerBlock(d_model, n_heads, 4*d_model)
                                       for _ in range(n_layers)])
        self.norm_f  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        pos = torch.arange(N, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)     # add PE

        mask = torch.triu(torch.ones(N, N), 1).bool().to(idx.device)
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.norm_f(x)
        return self.head(x)                           # logits over vocab
```

Train this on Tiny Shakespeare → working Shakespeare-in-the-style-of generator. Seriously.

---

# Why the Transformer won

1. **Parallelism** — unlike RNNs, all positions compute in parallel. GPUs love it.
2. **Long-range** — direct path between any two positions via attention, not $L$ steps of recurrence.
3. **Scaling laws** — performance keeps improving with more data, more params, more compute (Kaplan 2020; Chinchilla 2022 — L15).
4. **Transfer** — pre-train once, fine-tune everywhere (BERT, GPT pattern).
5. **Simplicity** — one block, stacked. Easier to build chips (TPUs, H100s) for.

---

<!-- _class: summary-slide -->

# Lecture 13 — summary

- **Transformer block** · MHA + FFN + residuals + pre-norm. Stack 10–100 times.
- **Multi-head attention** — h parallel heads, each a subspace; concat → output projection.
- **Positional encoding** — sinusoidal (original) · learned · **RoPE** (modern default in LLMs).
- **Three flavours** — encoder-only (BERT) · decoder-only (GPT, Llama, Claude) · encoder-decoder (T5).
- **Causal mask** — upper-triangular $-\infty$ prevents peeking ahead in autoregressive generation.
- **nanoGPT** — ~80 lines · the Transformer stack in PyTorch · trains on Tiny Shakespeare and generates passable pastiche.

### Read before Lecture 14

**Prince Ch 12 pretraining** sections.

### Next lecture

**Tokenization &amp; Pretraining Paradigms** — BPE from scratch, WordPiece, SentencePiece, BERT masked LM, GPT causal LM, T5 text-to-text.

<div class="notebook">

**Notebook 13** · `13-nanogpt.ipynb` — build the full Transformer block + nanoGPT from scratch; train on Tiny Shakespeare; generate text.

</div>
