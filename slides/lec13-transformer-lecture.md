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

# Learning outcomes

By the end of this lecture you will be able to:

1. List the **five ingredients** of a Transformer block and their roles.
2. Distinguish **pre-norm** from **post-norm** and know when each wins.
3. Compute **parameter count** for a block (attention + FFN).
4. Write **multi-head attention** in ~20 lines of PyTorch.
5. Choose a **positional encoding** scheme (sinusoidal / learned / RoPE / ALiBi).
6. Pick **encoder-only / decoder-only / enc-dec** for a given task.

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

# Pre-norm vs post-norm · a critical detail

Vaswani 2017's original was **post-norm** · `x = LayerNorm(x + Sublayer(x))`. Everyone uses **pre-norm** now · `x = x + Sublayer(LayerNorm(x))`.

<div class="keypoint">

Why the switch? Pre-norm keeps the residual path *unnormalized* — gradients flow through $x$ directly. Post-norm squashes gradient magnitude at every layer, which destabilizes training past ~12 layers.

</div>

Xiong et al. 2020 showed pre-norm trains without warmup and scales to 100+ layers. Post-norm needs careful warmup and usually breaks past 24. **Pre-norm is a free upgrade** — if you write your own Transformer, use pre-norm.

---

# The FFN · not an afterthought

Between each attention sublayer sits a two-layer MLP with a massive hidden size (4× d_model):

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x)$$

<div class="keypoint">

**~⅔ of Transformer parameters live in the FFN**, not in attention. Attention mixes tokens; the FFN transforms each token independently with huge capacity. Recent interpretability work (Anthropic) shows FFN layers store *facts* and *concepts*; attention layers route information between them.

</div>

GELU activation (smoother than ReLU) is the standard choice. Llama 2+ uses SwiGLU, a slightly better variant.

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

# Multi-head · the pipeline in detail

![w:920px](figures/lec13/svg/multi_head_detail.svg)

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

# The parameter accounting

For a Transformer with $d_\text{model} = 512$, $d_\text{ff} = 2048$, $h = 8$ heads:

<div class="math-box">

| Component | Params |
|:-:|:-:|
| $W_Q, W_K, W_V, W_O$ (attention) | $4 \cdot 512^2 = 1.05M$ |
| $W_1, W_2$ (FFN) | $2 \cdot 512 \cdot 2048 = 2.10M$ |
| LayerNorm × 2 | ~2k (negligible) |
| **Total per block** | **~3.15M** |

</div>

Note · attention params are **independent of sequence length** — the same weights process 10 tokens or 10,000. That's the big scaling advantage over RNNs, whose hidden state grows if you widen memory.

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

# Why sinusoidal · the clever part

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}), \quad PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

<div class="keypoint">

Different frequencies give a **multi-scale clock** · some dimensions wrap every 2 positions, others every 10,000. The model can read position information at whatever scale it needs.

</div>

Nicer property · $PE_{pos+k}$ is a linear transformation of $PE_{pos}$ (rotation in each 2-dim subspace). The model can learn to attend to *relative* positions ("3 steps to my left") using dot products of these embeddings — no need to memorize absolute positions.

Learned embeddings work too but don't extrapolate past training length. Sinusoidal does.

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

# Full stack · one figure

![w:920px](figures/lec13/svg/transformer_full_stack.svg)

---

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

# Cross-attention · for the encoder-decoder case

In the original Vaswani Transformer the *decoder* has **three** sublayers, not two:

1. **Self-attention** (causal) over target tokens generated so far.
2. **Cross-attention** · Q from decoder, K/V from encoder output. This is how decoder reads source.
3. **FFN** as usual.

<div class="keypoint">

Cross-attention is the Bahdanau-attention mechanism from L12, with learned Q/K/V projections. The encoder produces a rich representation of the source; the decoder queries it at every step.

</div>

GPT and Llama drop the encoder and cross-attention entirely — decoder-only. T5 keeps them for translation. Stable Diffusion uses cross-attention to inject text conditioning into images (L22).

---

# Common variations you will meet

<div class="math-box">

| Variation | Change | Seen in |
|:-:|:-:|:-:|
| **Pre-norm** (vs post-norm) | normalize before sublayer | GPT-2+, Llama, Claude |
| **SwiGLU FFN** (vs ReLU) | SiLU + gating | Llama 2+ |
| **RoPE** (vs sinusoidal PE) | rotate Q, K per position | Llama, Mistral, PaLM |
| **GQA** (vs MHA) | fewer KV heads than Q heads | Llama 2 70B+ |
| **RMSNorm** (vs LayerNorm) | drop mean centering | Llama, Mistral |
| **Parallel attention + FFN** | attn and FFN run in parallel, not sequentially | GPT-J, PaLM |

</div>

<div class="insight">

Each tweak is small (0.1-1% win). Stacked, they define a "2026 default Transformer" that looks quite different from Vaswani 2017 in details, identical in structure.

</div>

---

# Debug · "my Transformer doesn't train"

Top 5 issues to check:

1. **Learning rate too high** · with pre-norm no warmup is often fine; with post-norm, warmup of ≥100 steps is mandatory. Default lr = 3e-4 with AdamW β₂=0.95.
2. **Wrong attention mask** · forgot causal mask on a decoder? Model cheats during training, fails at inference.
3. **Embedding / output weight tied** · `self.head.weight = self.tok_emb.weight` reduces params by ~25%, usually helps.
4. **Float precision** · softmax overflows in fp16. Use bf16 or fp32 for the softmax.
5. **Positional encoding bug** · forgot to add PE? Or added twice? Position blind = garbage.

<div class="warning">

Karpathy's "most common deep-learning bug" list puts attention-mask bugs at the top. Every implementation has one that costs a week of debugging.

</div>

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
