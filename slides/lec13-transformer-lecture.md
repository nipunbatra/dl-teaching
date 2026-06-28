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

Last lecture (L12): **attention** fixed Seq2Seq's bottleneck — Q, K, V scaled dot product, softmax, weighted values. But in L11–L12 attention was *bolted onto an RNN* · the encoder still steps through tokens **one at a time**.

<div class="insight">

Attention works — but the RNN underneath is **inherently sequential**: token $t$ waits for token $t{-}1$, so it can't use a GPU's parallelism, and long-range info still crawls through many steps. The 2017 move · attention already lets any token see any other token directly, so **drop recurrence entirely** and keep only attention. "Attention Is All You Need." Today we stack that into the Transformer.

</div>

<div class="paper">

**Reading & inspiration** ·
- **Jay Alammar · *Illustrated Transformer*** + ***Illustrated GPT-2*** — the visual canon.
- **Karpathy · *Let's build GPT: from scratch, in code***  (YouTube) — full implementation walk-through.
- **Karpathy · `nanoGPT`** (GitHub) — clean reference implementation.
- **CS224N (Manning) · Lectures 8–9** — Stanford slides on Transformers.
- **UDL (Prince) · Ch 12 mid sections**.

</div>

---

# Four questions

1. What's in a full Transformer block — and why that order?
2. Why **multi-head** attention instead of one big head?
3. How does the model know position if there's no recurrence?
4. What's the difference between encoder, decoder, and decoder-only models (GPT)?

---

▶ **Interactives for L13** · [attention](https://nipunbatra.github.io/interactive-articles/attention/) (Q,K,V heatmap) · [positional-encoding](https://nipunbatra.github.io/interactive-articles/positional-encoding/) (sinusoid vs RoPE vs ALiBi side-by-side).

---

# Pop quiz · what's missing from "attention alone"?

L12 gave us attention · weighted lookup, $O(n^2)$ across the sequence.

<div class="popquiz">

If you stack 12 layers of pure self-attention, three problems appear ·

(a) The output of attention is a **linear combination of values** — no per-position non-linearity.
(b) Two sentences with the same words in different order get the **same** representation.
(c) Stacking 12 attention layers will have **vanishing/exploding** activations.

Stop and predict what the Transformer block adds to fix each.

</div>

Keep your three guesses — each Part of today fixes exactly one of them. We tally at the end.

---

<!-- _class: section-divider -->

### PART 1

# The block

What do you wrap around attention so it stacks 100 deep?

---

# The block · "communication then thinking"

<div class="keypoint">

A Transformer block is a **two-phase meeting** for your tokens.

**Phase 1 · communication** (attention) · every token looks at every other token to gather context.

**Phase 2 · thinking** (FFN) · each token individually processes what it heard, with no further mixing.

</div>

The block then repeats N times. By the end, each token has thought about every other token N times, with personal processing in between.

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

# Pre-norm vs post-norm · the gradient highway

<div class="insight">

**Analogy.** Gradient = a car driving back from the loss to the start of the network.
- **Post-norm** · toll booth (`LayerNorm`) on the *main* highway after every block. After 100 blocks, the car has barely any momentum left.
- **Pre-norm** · toll booth on an *off-ramp* (the sublayer). Main highway is clear · gradient speeds back unobstructed.

</div>

---

# ⭐⭐⭐ Optional · Pre-norm vs post-norm · derive the gradient

Goal · the block's input-to-output derivative $\partial x_\text{out} / \partial x_\text{in}$.

**Post-norm** · `x_out = LayerNorm(x_in + Sublayer(x_in))`. Gradient passes *through* LN:
$$\frac{\partial x_\text{out}}{\partial x_\text{in}} = \underbrace{\frac{\partial \text{LN}}{\partial(\cdot)}}_{\text{complex, < 1 typically}} \cdot \left(1 + \frac{\partial \text{Sub}}{\partial x_\text{in}}\right) \;\Rightarrow\; \text{shrink factors compound} \to \text{dies}$$

**Pre-norm** · `x_out = x_in + Sublayer(LayerNorm(x_in))`:
$$\frac{\partial x_\text{out}}{\partial x_\text{in}} = \mathbf{1} + \frac{\partial \text{Sub}(\text{LN}(x_\text{in}))}{\partial x_\text{in}}$$
The $\mathbf{1}$ is a clean identity — a direct path back. **Stable at any depth.**

Xiong et al. 2020 · pre-norm trains without warmup, scales to 100+ layers; post-norm needs careful LR warmup past ~24. **Use pre-norm.**

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

# ⭐⭐⭐ Optional · Multi-head · trace the tensor shapes

Setup · 1 sentence, 3 tokens, $d_\text{model} = 4$, $h = 2$ heads, so $d_k = 4/2 = 2$.

1. **Input** · `x.shape = (1, 3, 4)`.
2. **Project.** $W_Q, W_K, W_V$ are each $(4, 4)$.
$Q = xW_Q$ → `(1, 3, 4)`. Same for $K, V$.
3. **Split into heads.** Reshape last dim $4 \to (h=2, d_k=2)$, transpose to group by head:
$Q$: `(1, 3, 4)` → `(1, 3, 2, 2)` → `(1, 2, 3, 2)`. Same for $K, V$.
4. **Attention per head** (in parallel). Each head: `(3, 2)` → `(3, 2)`. Stack: `(1, 2, 3, 2)`.
5. **Concat + project.** Reverse the reshape → `(1, 3, 4)`. Final projection $W_O$ → `(1, 3, 4)`.

Output shape is **identical to input**. The block is composable — stack as many as you want.

---

# Multi-head · numeric example

$d_\text{model} = 4$, $h = 2$, $d_k = 2$. One token's projected $Q$ is $q = [1, 2, 3, 4]$.

**Split into 2 heads.** $q^{(1)} = [1, 2]$, $q^{(2)} = [3, 4]$.

Two other tokens' keys: $k_1 = [5, 6, 7, 8]$, $k_2 = [9, 0, 1, 2]$. Split into heads:
- $k_1^{(1)} = [5, 6]$, $k_1^{(2)} = [7, 8]$
- $k_2^{(1)} = [9, 0]$, $k_2^{(2)} = [1, 2]$

**Head 1 raw scores.** $q^{(1)} \cdot k_1^{(1)} = 5 + 12 = 17$, $q^{(1)} \cdot k_2^{(1)} = 9 + 0 = 9$. Head 1 prefers token 1.

**Head 2 raw scores.** $q^{(2)} \cdot k_1^{(2)} = 21 + 32 = 53$, $q^{(2)} \cdot k_2^{(2)} = 3 + 8 = 11$. Head 2 also prefers token 1, but with very different magnitude — they learn **different relationships**.

---

# Why split into heads?

A single attention head has to choose *one* distribution over positions per query. But real language has **multiple relations** to track at once:

- Syntax — who modifies whom
- Semantics — word meaning similarity
- Coreference — pronoun resolution
- Position — relative distance

<div class="keypoint">

Multi-head attention is a **team of specialists** — multiple attention circuits running in parallel, each specializing in one kind of relationship. The outputs are concatenated and projected; the network learns the division of labor.

</div>

Empirically, 8 or 16 heads is standard. Increasing beyond has diminishing returns — each head's dim $d_k = d / h$ gets too small to be useful.

---

# Parameter accounting · derive the formulas

Block params depend on $d_\text{model}$ and $d_\text{ff}$.

**Attention.** Four matrices, each $(d_\text{model}, d_\text{model})$:
- $W_Q, W_K, W_V$ project to QKV.
- $W_O$ mixes head outputs.
$$\text{Params}_\text{MHA} = 4\,d_\text{model}^2$$

**FFN.** Two layers · $d_\text{model} \to d_\text{ff} \to d_\text{model}$:
$$\text{Params}_\text{FFN} = d_\text{model} \cdot d_\text{ff} + d_\text{ff} \cdot d_\text{model} = 2\,d_\text{model}\,d_\text{ff}$$

**LayerNorm.** Each LN has scale + shift = $2 \cdot d_\text{model}$. Two LNs per block:
$$\text{Params}_\text{LN} = 4\,d_\text{model}$$

(Number of heads $h$ doesn't change the total — only how we partition $d_\text{model}$.)

---

# Worked numeric · where parameters live

$d_\text{model} = 512$, $d_\text{ff} = 2048$, $h = 8$:

| Component | Calculation | Params |
|:-:|:-:|:-:|
| Attention | $4 \cdot 512^2$ | **1,048,576** (~33%) |
| FFN | $2 \cdot 512 \cdot 2048$ | **2,097,152** (~66%) |
| LayerNorm × 2 | $4 \cdot 512$ | 2,048 (<0.1%) |
| **Total** | | **~3.15M** |

**Conclusion · the FFN ("thinking") uses 2× the parameters of attention ("communication").**

Attention params are **independent of sequence length** — the same weights process 10 or 10,000 tokens. Big scaling advantage over RNNs.

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

No recurrence — so how does the model know token order?

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

# Sinusoidal PE · the multi-handed clock analogy

<div class="insight">

Imagine encoding position with a **clock with many hands**.
- One hand ticks every position.
- Another every 10 positions.
- Another every 10,000.

Reading all hands gives a unique signature for each position. To get the signature for *position + 1*, you just rotate each hand a fixed amount — easy for the model to learn the relative offset.

</div>

Sinusoidal encoding is a high-dimensional version of this clock:
$$PE_{(pos, 2i)} = \sin(\theta_i \cdot pos),\quad PE_{(pos, 2i+1)} = \cos(\theta_i \cdot pos),\quad \theta_i = \frac{1}{10000^{2i/d}}$$

---

# ⭐⭐⭐ Optional · Why sinusoidal · derive the rotation property

For one pair of dimensions $(2i, 2i+1)$, the encoding at position $pos$ is $[\sin(\theta\,pos), \cos(\theta\,pos)]$.

What about $pos + k$? Trig identities:
- $\sin(a+b) = \sin a \cos b + \cos a \sin b$
- $\cos(a+b) = \cos a \cos b - \sin a \sin b$

Letting $a = \theta\,pos$, $b = \theta\,k$:
$$\begin{pmatrix} \sin\theta(pos+k) \\ \cos\theta(pos+k) \end{pmatrix} = \underbrace{\begin{pmatrix} \cos\theta k & \sin\theta k \\ -\sin\theta k & \cos\theta k \end{pmatrix}}_{\text{rotation matrix }R(k)} \begin{pmatrix} \sin\theta\,pos \\ \cos\theta\,pos \end{pmatrix}$$

**A 2D rotation matrix that depends only on $k$, not on $pos$!** The model can learn one linear transformation per relative offset → great at *relative* positions.

---

# ⭐⭐⭐ Optional · Rotation property · worked numeric

$pos = 5$, $k = 2$, $\theta = 0.1$.
- $PE_5 = [\sin 0.5, \cos 0.5] = [0.479, 0.878]$
- $PE_7 = [\sin 0.7, \cos 0.7] = [0.644, 0.765]$
- $R(2) = \begin{pmatrix} 0.980 & 0.199 \\ -0.199 & 0.980 \end{pmatrix}$
- $R(2) \cdot PE_5 = [0.980 \cdot 0.479 + 0.199 \cdot 0.878,\ -0.199 \cdot 0.479 + 0.980 \cdot 0.878] = [0.644, 0.765]$ ✓ matches $PE_7$.

Learned embeddings work too but don't extrapolate past training length. Sinusoidal does.

---

# The four positional encodings you'll meet

| Method | How | Used in |
|--------|-----|---------|
| **Sinusoidal** (Vaswani 2017) | fixed sin/cos | original Transformer, BERT |
| **Learned** | `nn.Embedding(max_len, d_model)` | GPT-2, GPT-3 |
| **RoPE** (Su 2021) | rotate Q and K by position-dependent angle | Llama, Mistral, PaLM, modern LLMs |
| **ALiBi** (Press 2021) | bias attention scores by relative distance | OPT-175B, some variants |

<div class="realworld">

As of this writing, **RoPE** dominates new LLMs. We'll cover it in L15 (LLMs). For now, any of the four works — pick the one that matches your base model.

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

The training objectives differ too · encoder-only predicts **masked tokens** (sees both sides), decoder-only predicts the **next token** (causal). Compared in depth in L14.

Currently, **decoder-only** dominates LLMs. Encoder-only ships in retrieval pipelines. Encoder-decoder survives for translation-style tasks.

---

# Causal masking · no peeking at the answer

Decoder predicting "The quick brown fox ___" (answer: "jumps"). During training the whole sentence is fed; when predicting position 5, the model must **not** see token 5.

**Causal mask** · add $-\infty$ to all entries above the diagonal *before* softmax:
$$\text{scores}_{i,j} = \begin{cases} QK^\top_{i,j}/\sqrt{d_k} & j \le i \\ -\infty & j > i \end{cases}$$
$\exp(-\infty) = 0$, so future positions get **exact zero** weight after softmax.

**Worked numeric.** Token 3's pre-softmax scores: $[1.0, 2.5, 0.5, 3.0]$. Without mask it would attend mostly to token 4 (score 3.0).

Apply mask → $[1.0, 2.5, 0.5, -\infty]$. Then:
$\exp = [2.72, 12.18, 1.65, 0]$, sum $= 16.55$.
Weights $= [0.16, 0.74, 0.10, \mathbf{0.00}]$.

Token 4 weight is exactly 0 — the cheat is closed off.

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

# Build GPT-tiny

Can ~80 lines of PyTorch really generate Shakespeare?

---

# The full stack · one figure

![w:920px](figures/lec13/svg/transformer_full_stack.svg)

---

<!-- _class: code-heavy -->

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

# Variations you will meet · structure & norm

<div class="math-box">

| Variation | Change | Seen in |
|:-:|:-:|:-:|
| **Pre-norm** (vs post-norm) | normalize before sublayer | GPT-2+, Llama, Claude |
| **RMSNorm** (vs LayerNorm) | drop mean centering | Llama, Mistral |
| **Parallel attention + FFN** | attn and FFN run in parallel, not sequentially | GPT-J, PaLM |

</div>

<div class="insight">

Each tweak is small (0.1-1% win). Stacked, they define the current default Transformer — quite different from Vaswani 2017 in details, identical in structure.

</div>

---

# Variations · component upgrades

<div class="math-box">

| Variation | Change | Seen in |
|:-:|:-:|:-:|
| **SwiGLU FFN** (vs ReLU) | SiLU + gating | Llama 2+ |
| **RoPE** (vs sinusoidal PE) | rotate Q, K per position | Llama, Mistral, PaLM |
| **GQA** (vs MHA) | fewer KV heads than Q heads | Llama 2 70B+ |
| **FlashAttention** | tile the $n\times n$ computation, never materialize it | every 2024+ training stack |

</div>

<div class="realworld">

RoPE, GQA, and FlashAttention all return in **L15** — they are what let modern LLMs scale to million-token contexts.

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

# Putting it all together · the L13 master sentence

<div class="math-box">

**A Transformer block = self-attention + FFN, each wrapped in residual + LayerNorm.** Stack 12 of those, add positional encodings + token embeddings, ship a final linear-to-vocab head. That's GPT. **The block *structure* has been stable since 2017** — the details (norm, PE, FFN) evolved, but mostly its scale changed.

</div>

| Block component | Role | Why it's there |
|:-:|:-:|:-:|
| Multi-head attention | mix tokens across positions | content-based routing |
| FFN | per-position non-linearity | non-linear transform |
| Residual + LN | gradient highway | trainable depth |
| Positional encoding | inject order | attention is permutation-equiv. |
| Causal mask (decoder) | prevent peeking ahead | autoregressive training |

Three flavours · **encoder-only** (BERT, classifying), **decoder-only** (GPT, generating), **encoder-decoder** (T5, translating).

---

# Pop quiz · revisit

The three problems with "attention alone"? Each Part fixed one:

<div class="keypoint">

(a) Attention output is linear in the values → the **FFN** adds the per-position non-linearity (Part 1).
(b) Same words, different order, same output → **positional encoding** breaks the tie (Part 3).
(c) 12 stacked layers blow up or die → **residual + LayerNorm** keep activations and gradients healthy (Part 1).

</div>

If you predicted all three, you effectively re-derived the Transformer block.

---

# Practice problems

<div class="math-box">

**P1.** A Transformer block has $d_\text{model}=768$ and FFN hidden $d_\text{ff}=3072$. Count parameters of (a) one MHA layer with 12 heads of $d_k=64$, (b) one FFN. Which dominates?

**P2.** Show that without positional encoding, a Transformer would output the **same logits** for "dog bites man" and "man bites dog." Where does the asymmetry need to come from?

**P3.** Explain the difference between **pre-LN** $x + \text{Attn}(\text{LN}(x))$ and **post-LN** $\text{LN}(x + \text{Attn}(x))$. Which is preferred for training stability and why?

</div>

---

# Practice problems · masks &amp; heads

<div class="math-box">

**P4.** A causal/decoder mask keeps only the lower triangle of the attention-score matrix. Which entries are set to $-\infty$, and why does this implement "no peeking"?

**P5.** Sketch how multi-head attention is implemented as a single matrix multiplication using reshape/permute. Why is this faster than running $h$ separate attention modules in parallel?

**P6.** Explain in one paragraph why GPT-2 is "just a bigger GPT-1" but BERT is structurally different. What is the single architectural switch?

</div>

---

# What breaks · symptom → suspect → test

| Symptom | Suspect | Fastest test |
|---|---|---|
| Loss stuck at $\ln(\text{vocab})$ from step 1 | forgot causal mask or positional encoding | check attention matrix is lower-triangular · diff outputs on a shuffled input |
| Teacher-forced loss superb, generated text garbage | mask leak — the model saw the future in training | measure next-token accuracy *autoregressively*, not teacher-forced |
| Loss diverges once you stack past ~24 layers | post-norm without warmup | switch to pre-norm, or add LR warmup and retry |
| NaN appears only in fp16 runs | softmax overflow in half precision | run softmax in fp32 / bf16, same seed, compare |
| Quality collapses past training context length | learned positional embeddings can't extrapolate | eval at 2× train length · swap in RoPE/ALiBi |

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

---

# The one-sentence takeaway

<div class="insight">

**The transformer is five ingredients, none of them new.**

</div>

*Next: before a transformer reads anything, someone has to chop text into tokens — and that choice haunts everything downstream.*
