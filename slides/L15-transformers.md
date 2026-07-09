---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Attention & Transformers II — the Transformer, Built Live

## Lecture 15 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Last time you built **one attention head**. Today you learn the **one block** that wraps it — and discover that every model in the rest of this course is *that block, stacked and scaled*.

<div class="keypoint">

**A Transformer block is: attention, then a small MLP — each on a residual highway, each preceded by a LayerNorm.** Nothing in it is new to you. The genius was *gluing five familiar pieces into a unit you can safely stack 100 deep.* By the end we build **nanoGPT** — the whole thing in ~80 lines — and watch it write Shakespeare.

</div>

You met attention — $Q,K,V$, $\text{softmax}(QK^\top/\sqrt{d_k})V$, the causal mask — in **L14**. Today we answer the question it left open: **what do you wrap around a head to get GPT?**

$$\underbrace{\text{one head (L14)}}_{\text{communicate}} \rightarrow \underbrace{\text{the block}}_{+\ \text{think, normalize, residual}} \rightarrow \underbrace{\text{multi-head}}_{\text{parallel subspaces}} \rightarrow \underbrace{\text{position}}_{\text{sinusoid / RoPE}} \rightarrow \underbrace{\text{nanoGPT}}_{\text{the payoff}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's block in:**
- **L16** — LLMs & scaling laws; RoPE, GQA, FlashAttention return here
- **L17** — alignment & fine-tuning ride *on top of* a decoder stack
- **L18** — vision-language: ViT is this block on image patches; cross-attention injects text
- **L21–L22** — diffusion U-Nets use attention; cross-attention carries the prompt
- **L23** — the KV-cache and attention's $O(n^2)$ cost decide inference speed

</div>
<div>

**Borrowed from earlier** (we build on, not repeat):
- attention · $Q,K,V$ · $\sqrt{d_k}$ · causal mask — **L14**
- residual connections — **L02 / L04**
- LayerNorm — **L07**
- token embeddings — **L13**

</div>
</div>

<div class="insight">

This is the **crown-jewel lecture**: one architecture that has been structurally stable since 2017, under every model you'll meet for the rest of the semester.

</div>

---

# Recall (L14): one attention head, in one line

A head produces, for each token, **one weighted average of the other tokens' value vectors** — the weights are normalized query–key similarities.

<div class="columns">
<div>

![w:420px](figures/lec13/svg/attention_weighted_average.svg)

</div>
<div>

<div class="math-box">

$$\text{Attn}(Q,K,V)=\underbrace{\text{softmax}\!\Big(\tfrac{QK^\top}{\sqrt{d_k}}\Big)}_{\text{weights, sum to }1}\,V$$

</div>

- $\sqrt{d_k}$ keeps scores in softmax's responsive range (L14).
- **Causal mask** = add $-\infty$ above the diagonal so a token can't peek ahead (L14).

</div>
</div>

That single operation is *all* we reuse. Everything today is the machinery around it. **Attention communicates; the rest of the block thinks, stabilizes, and stacks.**

---

<!-- _class: section-divider -->

## Part 1 · The Transformer block

---

# The block is a two-phase meeting for your tokens

<div class="keypoint">

**Phase 1 · communicate** (attention) — every token looks at every other token and gathers context.

**Phase 2 · think** (FFN) — each token individually processes what it heard, with **no** further mixing between tokens.

</div>

Then the block **repeats $N$ times**. After $N$ rounds, every token has communicated with every other token $N$ times, with private processing in between. That alternation — *mix, then transform* — is the entire design.

---

# The Transformer block (pre-norm), in one picture

![w:660px](figures/lec13/svg/transformer_block.svg)

Read it top-down: `LayerNorm → attention → add residual`, then `LayerNorm → FFN → add residual`. Input shape equals output shape, so blocks snap together like Lego.

---

# Why this exact structure? Three pieces you've already met

- **Attention** (L14) — the only part that mixes information *across* positions.
- **Residual connection** (L02/L04) — `x + sublayer(x)`; a gradient highway that makes depth trainable.
- **LayerNorm** (L07) — rescales each token vector so activations don't drift as you stack.

<div class="keypoint">

None of the three is new. The 2017 contribution was **assembling them** — attention for communication, an MLP for computation, residual + norm so the whole thing survives being stacked 100 layers deep. **BERT, GPT, Llama, Claude are all this block** — different depth, width, and data.

</div>

---

# The residual stream: the modern mental model

Because every sublayer is `x = x + f(x)`, the vector $x$ is never overwritten — each block **reads** from it and **adds** a correction back.

<div class="insight">

Think of $x$ as a **shared bus (the residual stream)** running the height of the network. Attention *writes* routed information onto it; the FFN *writes* transformed features onto it. Nothing is ever deleted — only added. This is why you can chop a trained Transformer's layers and it degrades gracefully, and why interpretability research reads features *off the stream*.

</div>

---

# Pre-norm vs post-norm — the gradient highway

<div class="insight">

**Analogy.** The gradient is a car driving back from the loss to the input.
- **Post-norm** — `LayerNorm(x + Sub(x))`: a toll booth on the *main* highway after every block. After 100 blocks the car has lost almost all its momentum → deep stacks starve.
- **Pre-norm** — `x + Sub(LayerNorm(x))`: the toll booth sits on an *off-ramp* (the sublayer). The main highway is clear → the gradient speeds straight back.

</div>

**Every modern LLM uses pre-norm** — it trains without learning-rate warmup and scales past 100 layers. (Gradient algebra in the appendix.)

---

# The FFN — not an afterthought

Between attention sublayers sits a two-layer MLP with a **4× wider** hidden dimension:

$$\text{FFN}(x)=W_2\,\big(\text{GELU}(W_1 x)\big),\qquad W_1:\ d_\text{model}\!\to\!4d_\text{model},\ \ W_2:\ 4d_\text{model}\!\to\!d_\text{model}$$

<div class="keypoint">

**~⅔ of a Transformer's parameters live in the FFN, not in attention.** Attention *routes* information between tokens; the FFN *transforms* each token independently, with huge capacity. Interpretability work suggests FFN layers store facts and concepts; attention layers move information between them.

</div>

GELU (a smooth ReLU) is the classic choice; Llama-family models swap in SwiGLU for a small win.

---

<!-- _class: code-heavy -->

# The block in PyTorch — 20 lines

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model),
        )

    def forward(self, x, mask=None):
        h = self.norm1(x)                       # pre-norm
        a, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + a                               # residual (communicate)
        x = x + self.ffn(self.norm2(x))         # residual (think)
        return x
```

<div class="code-note">

That is the whole Transformer. `x + ...` twice is the residual stream; `norm` before each sublayer is pre-norm; `mask` makes it a decoder. Everything else today is *plumbing around this class*.

</div>

---

<!-- _class: section-divider -->

## Part 2 · Multi-head attention

---

# One head is never enough — run several in parallel

![w:820px](figures/lec13/svg/multi_head_attention.svg)

Split $d_\text{model}$ into $h$ smaller subspaces, run an independent attention head in each, then concatenate and project back.

---

# Why split into heads?

A single head must commit to **one** distribution over positions per query. But language carries **several relations at once**:

- syntax — who modifies whom
- semantics — meaning similarity
- coreference — pronoun resolution
- position — relative distance

<div class="keypoint">

Multi-head attention is a **team of specialists**: $h$ attention circuits in parallel, each free to specialize in one kind of relationship. Their outputs are concatenated and mixed by $W_O$; the network *learns* the division of labor. Standard is $h=8$ or $16$ — beyond that each head's $d_k = d_\text{model}/h$ gets too small to be useful.

</div>

---

# The shape bookkeeping — split → attend → concat → project

![w:820px](figures/lec13/svg/multi_head_detail.svg)

Project with $W_Q,W_K,W_V$, **reshape** the last dim into $(h, d_k)$, attend **inside each head in parallel**, then **concat** the heads and apply $W_O$. Output shape $=$ input shape.

---

# Multi-head — a tiny numeric example

$d_\text{model}=4$, $h=2$, so $d_k=2$. One token's projected query is $q=[1,2,3,4]$.

**Split into 2 heads:** $q^{(1)}=[1,2]$, $q^{(2)}=[3,4]$.

Two other tokens' keys, also split: $k_1=[\underbrace{5,6}_{h1},\underbrace{7,8}_{h2}]$, $k_2=[\underbrace{9,0}_{h1},\underbrace{1,2}_{h2}]$.

- **Head 1 scores:** $q^{(1)}\!\cdot\! k_1^{(1)}=17$, $\ q^{(1)}\!\cdot\! k_2^{(1)}=9$ → prefers token 1, mildly.
- **Head 2 scores:** $q^{(2)}\!\cdot\! k_1^{(2)}=53$, $\ q^{(2)}\!\cdot\! k_2^{(2)}=11$ → prefers token 1, sharply.

Same tokens, **different relationships and magnitudes** — that is the point of separate subspaces.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1 · the shape trace.** A batch of $B=1$ sentence, $N=3$ tokens, $d_\text{model}=4$, $h=2$ heads. Trace the tensor shape through **project → split into heads → attend → concat → output-project**. What shape comes out, and why does that matter for stacking blocks?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

Start `x : (1, 3, 4)`. With $d_k = 4/2 = 2$:

1. **Project.** $W_Q,W_K,W_V$ are $(4,4)$; $\ Q=xW_Q \to (1,3,4)$ — same for $K,V$.
2. **Split into heads.** reshape $4\to(h{=}2,\,d_k{=}2)$, move head axis forward: $(1,3,4)\to(1,3,2,2)\to(1,2,3,2)$.
3. **Attend per head** (parallel). Each head: $(3,2)\to(3,2)$; stacked $(1,2,3,2)$.
4. **Concat + project.** reverse the reshape $\to (1,3,4)$; apply $W_O$ $\to (1,3,4)$.

<div class="keypoint">

**Output shape $=$ input shape $(1,3,4)$.** That invariance is *why blocks compose* — you can stack as many as you like without ever touching the shapes. Note $h$ only *partitions* $d_\text{model}$; it never changes the total width.

</div>

---

# Parameter accounting — derive the formulas

Block params depend on $d_\text{model}$ and $d_\text{ff}$ only.

<div class="columns">
<div>

**Attention** — four $(d_\text{model},d_\text{model})$ matrices $W_Q,W_K,W_V,W_O$:
$$P_\text{MHA}=4\,d_\text{model}^2$$

**LayerNorm** — scale + shift, two per block:
$$P_\text{LN}=4\,d_\text{model}$$

</div>
<div>

**FFN** — $d_\text{model}\!\to\! d_\text{ff}\!\to\! d_\text{model}$:
$$P_\text{FFN}=2\,d_\text{model}\,d_\text{ff}$$

With the usual $d_\text{ff}=4d_\text{model}$:
$$P_\text{FFN}=8\,d_\text{model}^2$$

</div>
</div>

<div class="insight">

So per block $\approx 4d^2 + 8d^2 = 12\,d_\text{model}^2$ — and the FFN is **twice** the attention. The head count $h$ is **absent**: it changes only *how* $d_\text{model}$ is partitioned, never the parameter total.

</div>

---

# Worked numeric — where the parameters live

$d_\text{model}=512$, $d_\text{ff}=2048$, $h=8$:

| Component | Calculation | Params |
|:-:|:-:|:-:|
| Attention | $4\cdot 512^2$ | **1.05 M** (~33%) |
| FFN | $2\cdot 512\cdot 2048$ | **2.10 M** (~66%) |
| LayerNorm × 2 | $4\cdot 512$ | 2 048 (<0.1%) |
| **Total / block** | | **≈ 3.15 M** |

Attention params are **independent of sequence length** — the same weights process 10 or 10 000 tokens. That is a decisive scaling advantage over RNNs.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2 · parameter count.** A GPT-2-scale block has $d_\text{model}=768$, $h=12$ heads ($d_k=64$), FFN width $d_\text{ff}=3072$. Count the parameters of **(a)** the multi-head attention and **(b)** the FFN. Which dominates, and does the answer change if you use 6 heads instead of 12?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

$$P_\text{MHA}=4\,d_\text{model}^2 = 4\cdot 768^2 = \mathbf{2.36\,M}$$
$$P_\text{FFN}=2\,d_\text{model}\,d_\text{ff} = 2\cdot 768\cdot 3072 = \mathbf{4.72\,M}$$

<div class="keypoint">

**(b) the FFN dominates** — exactly $2\times$ the attention (because $d_\text{ff}=4d_\text{model}$). Switching 12 heads → 6 heads changes **nothing**: $P_\text{MHA}=4d_\text{model}^2$ has no $h$ in it. Fewer heads just means each head is *wider* ($d_k=128$), same total. This is why "how many heads?" is a *modeling* choice, not a *budget* choice.

</div>

---

<!-- _class: code-heavy -->

# Multi-head attention by hand

```python
def multi_head_attention(x, Wq, Wk, Wv, Wo, n_heads):
    B, N, d = x.shape
    d_k = d // n_heads
    # project, then split heads: (B, N, d) -> (B, n_heads, N, d_k)
    q = (x @ Wq).view(B, N, n_heads, d_k).transpose(1, 2)
    k = (x @ Wk).view(B, N, n_heads, d_k).transpose(1, 2)
    v = (x @ Wv).view(B, N, n_heads, d_k).transpose(1, 2)
    # scaled dot-product attention, in parallel across heads
    scores  = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    weights = scores.softmax(dim=-1)
    out = (weights @ v).transpose(1, 2).contiguous().view(B, N, d)  # concat
    return out @ Wo                                                  # project
```

<div class="code-note">

All $h$ heads are one batched matmul — no Python loop. The `view/transpose` *is* the "split into heads"; the reverse `transpose/view` *is* the concat. In practice you'd call `F.scaled_dot_product_attention` (FlashAttention under the hood, L16).

</div>

---

# The cost of communication — attention is $O(n^2)$

The score matrix $QK^\top$ is $n\times n$: every token compares with every other. Memory and compute grow **quadratically** in sequence length $n$.

<div class="columns">
<div>

<div class="realworld">

Double the context, quadruple the attention cost. This single fact drives most modern LLM engineering — **FlashAttention** (never materialize the $n^2$ matrix), **KV-caching**, sliding-window and sparse attention. All return in **L16 / L23**.

</div>

</div>
<div>

<div class="notebook">

**🎛 Interactive · Attention, Calculated** — click a token, watch its embedding become $q,k,v$; inspect the full $n\times n$ matrix live (score → softmax → weighted $V$) and train a tiny head in-browser. *(Interactive Lab · `interactive/attention`)*

</div>

</div>
</div>

---

<!-- _class: section-divider -->

## Part 3 · Positional encoding

---

# The problem — attention is permutation-invariant

Attention computes $QK^\top$ — a set of pairwise similarities with **no notion of *where* each token sits**.

> "Dog bites man"  →  the *same* attention weights as  "Man bites dog"

<div class="keypoint">

Same tokens, reordered → identical representation. Attention alone is **order-blind**. We must **inject position** into the token embeddings before the first block, or the model can never tell these two sentences apart.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3 · why now, but not before?** An RNN reads "dog bites man" left-to-right and *never* needs a positional encoding, yet a Transformer breaks without one. **Why?** What structural property of the RNN silently supplies order — and what did the Transformer trade away to gain parallelism?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="insight">

An RNN is **sequential by construction**: it consumes token $t$ *after* token $t{-}1$, so order is baked into the *timing* of the computation — position is implicit in *when* each token is processed. The Transformer deliberately **threw recurrence away** to process all tokens **in parallel** on a GPU. That parallelism is exactly what removes the built-in sense of order.

</div>

<div class="keypoint">

**Position must be added back as data**, since it's no longer supplied by the computation. Parallelism and order-awareness are traded against each other — positional encoding buys back the order for a tiny additive cost.

</div>

---

# Sinusoidal positional encoding — a multi-scale clock

![w:760px](figures/lec13/svg/positional_encoding.svg)

<div class="columns">
<div>

$$PE_{(pos,2i)}=\sin(\theta_i\, pos),\quad PE_{(pos,2i+1)}=\cos(\theta_i\, pos)$$
$$\theta_i=1/10000^{\,2i/d}$$

</div>
<div>

Like a **clock with many hands** — one ticks every position, one every 10, one every 10 000. Reading all hands gives each position a unique signature; a shift of $+1$ just *rotates* every hand by a fixed angle, so **relative** offsets are easy to learn.

</div>
</div>

The vector is **added** to the token embedding (not concatenated) — same width, free to stack.

---

# RoPE, and the four encodings you'll meet

**RoPE** (rotary) doesn't add anything — it **rotates** $q$ and $k$ by an angle proportional to their position, so the dot product $q_i\!\cdot\!k_j$ ends up depending only on the **relative** offset $i-j$. That relative-ness is why it extrapolates and dominates modern LLMs.

| Method | How | Used in |
|--------|-----|---------|
| **Sinusoidal** (Vaswani '17) | fixed sin/cos, added | original Transformer, BERT |
| **Learned** | `nn.Embedding(max_len, d)` | GPT-2, GPT-3 |
| **RoPE** (Su '21) | rotate $Q,K$ by position angle | **Llama, Mistral, most new LLMs** |
| **ALiBi** (Press '21) | linear distance penalty on scores | some long-context variants |

<div class="realworld">

RoPE, ALiBi, and long-context extrapolation get their full treatment in **L16**. For today: any of the four works — pick what your base model uses.

</div>

---

# Explore position yourself

<div class="notebook">

**🎛 Interactive · Positional Encodings — Sinusoid, RoPE, ALiBi** — a sinusoidal heatmap with live sliders for $n$ and $d$; RoPE shown as a 2-D rotation where the dot product depends only on relative offset; an extrapolation panel (train at 2k, test at 32k) where sinusoidal collapses, RoPE drifts, ALiBi holds. *(Interactive Lab · `interactive/positional-encoding`)*

</div>

<div class="notebook">

**🎛 Interactive · the permutation demo** — shuffle a sentence's tokens with PE off, watch the logits stay identical; turn PE on, watch them diverge. The order-blindness of Part 3, made visible. *(bundled in `interactive/attention`)*

</div>

---

<!-- _class: section-divider -->

## Part 4 · Three flavours — BERT · GPT · T5

---

# One block, three ways to wire it

| Model | What it is | Attention | Use case |
|-------|------------|-----------|----------|
| **Encoder-only** (BERT) | stack of blocks, **no mask** | bidirectional | classification, embedding, retrieval |
| **Decoder-only** (GPT, Llama, Claude) | stack of blocks, **causal mask** | left-to-right | autoregressive generation |
| **Encoder-decoder** (T5, BART) | both + **cross-attention** | enc bi-, dec causal | translation, summarization |

<div class="keypoint">

The **only architectural switch** between BERT and GPT is the **causal mask**. Same block, same FFN, same residual stream — one lets a token see the future, the other doesn't. That single bit decides "understand" vs "generate."

</div>

---

# Encoder-decoder — the original 2017 design

![w:720px](figures/lec13/svg/full_transformer.svg)

The **encoder** reads the whole source bidirectionally; the **decoder** generates the target left-to-right, querying the encoder through cross-attention. GPT keeps only the right half.

---

# Decoder-only is today's default — and why

Same block; the difference is the **training objective**:

<div class="columns">
<div>

**Encoder-only (BERT)** — masked-token prediction. Blank out 15% of tokens, predict them from **both sides**. Great representations; can't generate.

</div>
<div>

**Decoder-only (GPT)** — next-token prediction. Predict token $t{+}1$ from tokens $1..t$ under a **causal mask**. One objective, infinitely much unlabeled text, and it *generates*.

</div>
</div>

<div class="realworld">

Decoder-only won because next-token prediction is a **universal, self-supervised** objective that scales with raw internet text — and the same trained model both understands and generates. Encoder-only survives in retrieval; encoder-decoder survives for translation-style tasks.

</div>

---

# The causal mask — no peeking at the answer

<div class="insight">

A decoder predicts the next token, so token $i$ may attend only to tokens $1\ldots i$. Enforce it by adding $-\infty$ to the upper triangle of the scores **before** softmax; $\exp(-\infty)=0$ zeroes those weights.

</div>

![w:560px](figures/lec13/svg/causal_mask.svg)

**Numeric.** Token 3's scores $[1.0,2.5,0.5,\,3.0]$ → mask position 4 → $[1.0,2.5,0.5,\,-\infty]$ → softmax $[0.16,0.74,0.10,\mathbf{0.00}]$. The future token gets **exactly zero** weight.

---

# Cross-attention — how a decoder reads the source

In the encoder-decoder, the decoder block has **three** sublayers, not two:

1. **Self-attention** (causal) over target tokens generated so far.
2. **Cross-attention** — $Q$ from the decoder, $K,V$ from the **encoder output**. This is how it reads the source.
3. **FFN**, as usual.

<div class="keypoint">

Cross-attention is the same operation as self-attention, only $K,V$ come from *another* sequence. It is how the decoder queries the encoder — and later, how **Stable Diffusion injects a text prompt into an image** (L22). GPT/Llama drop it entirely; T5 keeps it.

</div>

---

<!-- _class: section-divider -->

## Part 5 · nanoGPT, built live

---

# The full GPT stack — one figure

![w:760px](figures/lec13/svg/transformer_full_stack.svg)

Token embeddings **+** positional embeddings → $N$ decoder blocks (causal) → final LayerNorm → a linear head to vocab logits. That is the entire model.

---

# From tokens to the first hidden state

Before the blocks, two lookup tables build the input to the residual stream:

$$x = \underbrace{\text{tok\_emb}[\,\text{idx}\,]}_{\text{what the token is}} \;+\; \underbrace{\text{pos\_emb}[\,0,1,\dots,N{-}1\,]}_{\text{where it sits}}$$

<div class="insight">

Both are $\;(\cdot)\to d_\text{model}$ tables; you **add** them so the first block sees content and position on the *same* vector. At the very top, a linear **unembedding** maps the stream back to one logit per vocabulary token — often **weight-tied** to `tok_emb` to save ~25% of parameters.

</div>

---

<!-- _class: code-heavy -->

# nanoGPT — the model, ~30 lines

```python
class GPT(nn.Module):
    def __init__(self, vocab, d_model=192, n_heads=6, n_layers=6, max_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab,   d_model)      # what
        self.pos_emb = nn.Embedding(max_len, d_model)      # where
        self.blocks  = nn.ModuleList(
            TransformerBlock(d_model, n_heads, 4 * d_model) for _ in range(n_layers))
        self.norm_f  = nn.LayerNorm(d_model)
        self.head    = nn.Linear(d_model, vocab, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)          # embed + position
        mask = torch.triu(torch.ones(N, N, device=idx.device), 1).bool()
        for block in self.blocks:
            x = block(x, mask=mask)                         # N causal blocks
        return self.head(self.norm_f(x))                    # logits over vocab
```

<div class="code-note">

Reuses the 20-line `TransformerBlock` verbatim. The only decoder-specific line is the upper-triangular `mask`. `d_ff = 4 * d_model` is the standard FFN width. That's it — a complete GPT.

</div>

---

<!-- _class: code-heavy -->

# nanoGPT — the training loop

```python
model = GPT(vocab_size)
opt   = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))

for step in range(5000):
    xb, yb = get_batch("train")            # xb, yb : (B, N) tokens & next-tokens
    logits = model(xb)                     # (B, N, vocab)
    loss   = F.cross_entropy(logits.flatten(0, 1), yb.flatten())
    opt.zero_grad(); loss.backward(); opt.step()
```

<div class="code-note">

The loss is **plain cross-entropy** — exactly the negative log-likelihood from **L01**, one term per position, predicting the next token. Every position in the sequence is a training example *simultaneously*, thanks to the causal mask. No new machinery: it's next-token classification, at scale.

</div>

---

<!-- _class: code-heavy -->

# nanoGPT — generation (the payoff)

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, block_size, temp=1.0):
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])   # crop to the context window
        logits = logits[:, -1, :] / temp       # last position, temperature
        probs  = F.softmax(logits, dim=-1)
        nxt    = torch.multinomial(probs, 1)    # sample one token
        idx    = torch.cat([idx, nxt], dim=1)   # append, then repeat
    return idx
```

<div class="code-note">

Autoregression in five lines: predict → sample → append → feed back. `temp<1` sharpens (greedy-ish), `temp>1` flattens (creative) — the softmax-temperature knob from **L13**. Trained on ~1 MB of Tiny Shakespeare, this generates passable Shakespearean pastiche. **~80 lines total, no new ideas since L01.**

</div>

---

# Build it yourself

<div class="notebook">

**📓 Notebook · nanoGPT from scratch** — assemble `TransformerBlock` → `GPT`, train on Tiny Shakespeare, and sample with temperature. Watch the loss fall and the samples turn from noise into iambic-sounding lines. *(ES 667 · `notebooks/13-nanogpt.ipynb`)*

</div>

<div class="notebook">

**🎬 Karpathy · "Let's build GPT: from scratch, in code, spelled out"** (YouTube) — the definitive 2-hour build; this lecture is its condensed map. Reference implementation: **`github.com/karpathy/nanoGPT`** (MIT). *(A. Karpathy, Neural Networks: Zero to Hero)*

</div>

---

# Variations you'll meet — and where they return

<div class="math-box">

| Variation | Change vs 2017 | Seen in | Returns |
|:-:|:-:|:-:|:-:|
| **Pre-norm** | normalize before sublayer | GPT-2+, Llama, Claude | today |
| **RMSNorm** | drop mean-centering in LN | Llama, Mistral | L16 |
| **RoPE** | rotate $Q,K$ by position | Llama, Mistral, PaLM | L16 |
| **SwiGLU FFN** | gated SiLU activation | Llama 2+ | L16 |
| **GQA** | fewer KV heads than Q heads | Llama-2-70B+ | L16 |
| **FlashAttention** | never materialize the $n^2$ matrix | every 2024+ stack | L16 / L23 |

</div>

<div class="insight">

Each tweak is a **0.1–1% win**; stacked, they define today's default Transformer. The *structure* — attention + FFN + residual + norm — is **unchanged since 2017**. Only the components and the scale moved.

</div>

---

# This is the architecture behind everything next

<div class="keypoint">

**Every model in the rest of this course is this block, stacked and scaled.** GPT and Llama are decoder-only stacks; BERT flips off the mask; ViT (L18) runs it on image patches; diffusion U-Nets (L21–22) attend inside the denoiser; cross-attention carries prompts into images. Learn this one block and you have read the blueprint for the entire modern field.

</div>

The reason it won: **parallelism** (all positions at once), **long-range** (any token reaches any other in one hop), **scaling laws** (more data/compute keeps helping — L16), and **simplicity** (one block, easy to build chips for).

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**A Transformer block is attention + FFN, each on a residual highway behind a pre-norm; stack it with token+position embeddings and a causal mask, and you have GPT.**

</div>

- **The block** · communicate (attention) → think (FFN), each `x + sublayer(norm(x))`.
- **Multi-head** · $h$ parallel subspaces; params $\approx 12\,d_\text{model}^2$ per block, FFN dominates, $h$ irrelevant to the total.
- **Position** · attention is order-blind → add sinusoidal / RoPE; an RNN needs neither because it's sequential.
- **Three flavours** · BERT (no mask) · GPT (causal mask) · T5 (+cross-attention) — the mask is the switch.
- **nanoGPT** · ~80 lines, plain cross-entropy, and it writes Shakespeare.

**Next (L16):** from *one* Transformer to *large* language models — scaling laws, RoPE, GQA, FlashAttention, and what emerges at scale.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **nanoGPT, "Let's build GPT," the ~80-line build** — A. Karpathy, `github.com/karpathy/nanoGPT` (MIT) & *Neural Networks: Zero to Hero* (YouTube).
- **The block / multi-head / positional-encoding visual intuition** — J. Alammar, *The Illustrated Transformer* & *The Illustrated GPT-2*.
- **Transformer block, multi-head, positional encoding, three flavours, nanoGPT figures** — *ES 667 Deep Learning* transformer materials, N. Batra · `github.com/nipunbatra/dl-teaching`.
- **Pedagogical framing (sequence models → attention → Transformers)** — A. Ng, *Deep Learning Specialization*, Course 5 Week 4.
- **The architecture** — Vaswani et al., *Attention Is All You Need* (2017); pre-norm: Xiong et al. (2020); RoPE: Su et al. (2021).

Interactive Lab · `interactive/attention`, `interactive/positional-encoding` (nipunbatra.github.io/interactive-articles). Figures adapted from the ES 667 figure library. All source courses © N. Batra &amp; teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · pre-norm vs post-norm — derive the gradient

**Claim.** Pre-norm leaves a clean $+1$ identity path for the gradient; post-norm multiplies by a shrinking factor at every block, so deep stacks starve.

**Post-norm** · $x_\text{out}=\text{LN}\big(x_\text{in}+\text{Sub}(x_\text{in})\big)$. The gradient passes *through* LN:
$$\frac{\partial x_\text{out}}{\partial x_\text{in}}=\underbrace{\frac{\partial \text{LN}}{\partial(\cdot)}}_{\text{typically }<1}\Big(1+\frac{\partial \text{Sub}}{\partial x_\text{in}}\Big)\ \Rightarrow\ \text{factors }<1\text{ compound over depth}\to\text{vanish}.$$

**Pre-norm** · $x_\text{out}=x_\text{in}+\text{Sub}\big(\text{LN}(x_\text{in})\big)$:
$$\frac{\partial x_\text{out}}{\partial x_\text{in}}=\mathbf{1}+\frac{\partial\,\text{Sub}(\text{LN}(x_\text{in}))}{\partial x_\text{in}}.$$

The $\mathbf{1}$ is an unattenuated path back through *every* block — the residual stream as a gradient highway. Pre-norm trains without warmup and scales past 100 layers (Xiong et al. 2020). **Use pre-norm.**

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · why sinusoidal — the rotation property

**Claim.** A shift by $k$ positions *rotates* the encoding by a fixed angle that depends on $k$ but not on $pos$ — so the model learns one transform per relative offset.

For a dimension pair $(2i,2i{+}1)$, $PE_{pos}=[\sin(\theta pos),\ \cos(\theta pos)]$. Using the angle-sum identities with $a=\theta pos$, $b=\theta k$:

$$\begin{pmatrix}\sin\theta(pos{+}k)\\ \cos\theta(pos{+}k)\end{pmatrix}=\underbrace{\begin{pmatrix}\cos\theta k & \sin\theta k\\ -\sin\theta k & \cos\theta k\end{pmatrix}}_{R(k)\ \text{— depends only on }k}\begin{pmatrix}\sin\theta pos\\ \cos\theta pos\end{pmatrix}.$$

**Numeric check** ($pos{=}5,\ k{=}2,\ \theta{=}0.1$): $PE_5=[0.479,0.878]$, $PE_7=[0.644,0.765]$, and $R(2)\,PE_5=[0.644,0.765]=PE_7$ ✓.

Because the map is position-independent, relative offsets are trivial to represent — and sinusoidal extrapolates past training length where a *learned* table cannot. This same "rotate to encode relative position" idea, applied directly to $Q$ and $K$, **is RoPE** (L16).
