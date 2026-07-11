---
title: "Lecture 16: Transformers I"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 16: Transformers I  
## Self-Attention, Positional Encoding and Transformer Blocks

Core story:

\[
\boxed{
\text{Replace recurrence with attention between all tokens.}
}
\]

The Transformer uses attention mechanisms instead of recurrence/convolution, making sequence modelling much more parallelizable during training.

---

# Part I — Why Transformers?

### Slide 1 — Title

**Transformers I**

> Self-attention, positional encoding and Transformer blocks

---

### Slide 2 — Recall: RNN seq2seq with attention

Encoder:

\[
h_t=f(h_{t-1},x_t)
\]

Decoder attention:

\[
c_u=\sum_t\alpha_{ut}h_t
\]

Problem:

\[
h_t\text{ is still computed sequentially.}
\]

---

### Slide 3 — Transformer idea

Instead of:

\[
h_t=f(h_{t-1},x_t)
\]

use:

\[
h_t=f(x_1,\ldots,x_T)
\]

where each token can directly attend to other tokens.

\[
\boxed{
\text{self-attention relates positions within the same sequence}
}
\]

---

### Slide 4 — RNN vs CNN vs Transformer

| Model | Context mechanism | Parallel over positions? |
|---|---|---:|
| RNN | hidden state | no |
| CNN/TCN | receptive field | yes |
| Transformer | attention over tokens | yes |

---

# Part II — Attention as soft lookup

### Slide 5 — Attention from previous lecture

Query:

\[
q
\]

Keys:

\[
k_1,\ldots,k_T
\]

Values:

\[
v_1,\ldots,v_T
\]

Scores:

\[
s_i=q^\top k_i
\]

Weights:

\[
\alpha_i=\operatorname{softmax}(s)_i
\]

Output:

\[
o=\sum_i\alpha_i v_i.
\]

---

### Slide 6 — Query-key-value intuition

\[
q:\text{what am I looking for?}
\]

\[
k:\text{what do I contain?}
\]

\[
v:\text{what information do I provide?}
\]

Attention is differentiable retrieval.

---

### Slide 7 — Worked attention example

Let:

\[
q=[1,0]
\]

\[
k_1=[1,0],\quad k_2=[0,1],\quad k_3=[1,1]
\]

Scores:

\[
s=[1,0,1].
\]

Softmax:

\[
\alpha\approx[0.422,0.155,0.422].
\]

If:

\[
v_1=[10,0],\ v_2=[0,10],\ v_3=[10,10],
\]

then:

\[
o\approx[8.44,5.77].
\]

---

### Slide 8 — Scaled dot-product attention

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

This is the central Transformer operation.

---

### Slide 9 — Why scale by \(\sqrt{d_k}\)?

If:

\[
q_i,k_i\sim(0,1)
\]

then:

\[
q^\top k=\sum_{i=1}^{d_k}q_ik_i
\]

has variance approximately:

\[
d_k.
\]

Large \(d_k\) makes logits large, softmax sharp, gradients small.

So use:

\[
\frac{q^\top k}{\sqrt{d_k}}.
\]

---

### Slide 10 — Matrix shapes

For sequence length \(T\), model dimension \(d\):

\[
X\in\mathbb R^{T\times d}.
\]

Projection matrices:

\[
W_Q,W_K,W_V\in\mathbb R^{d\times d_k}.
\]

Then:

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
\]

\[
Q,K,V\in\mathbb R^{T\times d_k}.
\]

Attention matrix:

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
\in\mathbb R^{T\times T}.
\]

Output:

\[
O=AV\in\mathbb R^{T\times d_k}.
\]

---

# Part III — Self-attention

### Slide 11 — Self-attention

In self-attention, all \(Q,K,V\) come from the same sequence:

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V.
\]

Each token builds a new representation by mixing information from other tokens in the same sequence.

---

### Slide 12 — Attention matrix

\[
A_{ij}
=
\text{how much token }i\text{ attends to token }j.
\]

Rows sum to one:

\[
\sum_jA_{ij}=1.
\]

Visual: \(T\times T\) heatmap.

---

### Slide 13 — Worked self-attention mini-example

Take three token representations:

\[
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix}.
\]

Use:

\[
W_Q=W_K=W_V=I.
\]

Then:

\[
Q=K=V=X.
\]

Compute:

\[
QK^\top=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}.
\]

Then scale by:

\[
\sqrt{2}.
\]

---

### Slide 14 — Interpret the attention matrix

For row 1:

\[
[1,0,1]/\sqrt2.
\]

Token 1 attends more to tokens 1 and 3 than token 2.

After softmax, the output is a weighted average of value vectors.

---

### Slide 15 — Self-attention is content-dependent

In CNNs, receptive field pattern is fixed.

In self-attention:

\[
A_{ij}
\]

depends on token representations.

So each token can decide which other tokens matter.

---

### Slide 16 — Interactive 1: self-attention matrix

`self-attention-matrix.html`

Controls:

- token vectors;
- \(W_Q,W_K,W_V\);
- scaling on/off;
- selected token.

Show:

- \(QK^\top\);
- softmax attention matrix;
- output vectors;
- attention heatmap.

---

# Part IV — Masked / causal self-attention

### Slide 17 — Future leakage

For language modelling, token \(t\) must not see:

\[
x_{t+1},x_{t+2},\ldots
\]

So attention must be causal.

---

### Slide 18 — Causal mask

Before softmax:

\[
S=\frac{QK^\top}{\sqrt{d_k}}.
\]

Set:

\[
S_{ij}=-\infty
\quad\text{if }j>i.
\]

Then:

\[
A_{ij}=0
\quad\text{for }j>i.
\]

---

### Slide 19 — Causal mask matrix

For \(T=5\):

\[
\begin{bmatrix}
0&-\infty&-\infty&-\infty&-\infty\\
0&0&-\infty&-\infty&-\infty\\
0&0&0&-\infty&-\infty\\
0&0&0&0&-\infty\\
0&0&0&0&0
\end{bmatrix}
\]

Visual: lower-triangular attention.

---

### Slide 20 — Encoder vs decoder attention

Encoder self-attention:

\[
\text{can attend bidirectionally}
\]

Decoder self-attention:

\[
\text{must be causal}
\]

For GPT-style language modelling, use causal self-attention.

For BERT-style encoding, use bidirectional self-attention.

---

### Slide 21 — Interactive 2: causal attention

`causal-attention.html`

Controls:

- sequence length;
- selected query token;
- causal mask on/off;
- padding mask on/off.

Show:

- allowed positions;
- masked scores;
- attention weights.

---

# Part V — Multi-head attention

### Slide 22 — Why multiple heads?

One attention operation may focus on one relation.

Multiple heads allow different learned projections:

\[
\text{head}_i
=
\operatorname{Attention}
(XW_i^Q,XW_i^K,XW_i^V).
\]

Then concatenate:

\[
\operatorname{MHA}(X)
=
[\text{head}_1;\ldots;\text{head}_H]W_O.
\]

---

### Slide 23 — Multi-head shapes

Let:

\[
d_{\text{model}}=512,
\quad
H=8.
\]

Then usually:

\[
d_k=d_v=64.
\]

Each head works in a smaller subspace.

Concatenation returns:

\[
T\times512.
\]

---

### Slide 24 — What heads can learn

Possible relations:

- previous word;
- subject–verb relation;
- matching brackets;
- local phrase structure;
- position-based patterns;
- semantic similarity.

Caution: individual head interpretation can be noisy.

---

### Slide 25 — Worked MHA parameter count

For:

\[
d=512,\quad H=8.
\]

Using combined projections:

\[
W_Q,W_K,W_V,W_O\in\mathbb R^{512\times512}.
\]

Total parameters:

\[
4\cdot512^2
=
1{,}048{,}576.
\]

Ignoring biases.

---

### Slide 26 — Interactive 3: multi-head attention

`multi-head-attention.html`

Show:

- several heads;
- each head’s attention matrix;
- concatenated output;
- output projection;
- effect of changing one head.

---

# Part VI — Positional information

### Slide 27 — Self-attention alone is permutation equivariant

Without positional information:

\[
\text{Transformer}([x_1,x_2,x_3])
\]

does not inherently know which token is first, second, third.

The attention computation depends on content, not order.

---

### Slide 28 — Add positional encoding

Input representation:

\[
r_t=e_t+p_t
\]

where:

- \(e_t\): token embedding;
- \(p_t\): position embedding/encoding.

---

### Slide 29 — Learned positional embeddings

Learn a table:

\[
P\in\mathbb R^{T_{\max}\times d}.
\]

Position \(t\) gets:

\[
p_t=P_{t,:}.
\]

Simple and common.

Limitation:

\[
T>T_{\max}
\]

is not naturally handled.

---

### Slide 30 — Sinusoidal positional encoding

Original Transformer used sinusoidal encodings:

\[
PE_{(pos,2i)}
=
\sin\left(
\frac{pos}{10000^{2i/d}}
\right)
\]

\[
PE_{(pos,2i+1)}
=
\cos\left(
\frac{pos}{10000^{2i/d}}
\right).
\]

This gives deterministic position vectors.

---

### Slide 31 — Why many frequencies?

Low-frequency dimensions vary slowly.

High-frequency dimensions vary rapidly.

Together, they encode position at multiple scales.

Visual: sine/cosine waves across positions.

---

### Slide 32 — Worked position vector

For small \(d=4\):

\[
PE(pos)=
\left[
\sin\left(\frac{pos}{10000^0}\right),
\cos\left(\frac{pos}{10000^0}\right),
\sin\left(\frac{pos}{10000^{1/2}}\right),
\cos\left(\frac{pos}{10000^{1/2}}\right)
\right].
\]

At:

\[
pos=0
\]

\[
PE(0)=[0,1,0,1].
\]

---

### Slide 33 — Interactive 4: positional encodings

`positional-encoding.html`

Controls:

- position;
- model dimension;
- learned vs sinusoidal;
- compare two positions.

Show:

- position vectors;
- similarity matrix \(P P^\top\);
- sine/cosine components.

---

# Part VII — Transformer block

### Slide 34 — Block components

A Transformer block usually has:

1. multi-head self-attention;
2. residual connection;
3. normalization;
4. feed-forward network;
5. residual connection;
6. normalization.

---

### Slide 35 — Feed-forward network

Applied independently at each position:

\[
\operatorname{FFN}(x)
=
W_2\phi(W_1x+b_1)+b_2.
\]

Same FFN weights are shared across sequence positions.

Original Transformer used:

\[
d_{\text{ff}}=2048
\]

for:

\[
d_{\text{model}}=512.
\]

---

### Slide 36 — Why FFN after attention?

Self-attention mixes information across positions.

FFN transforms each position’s representation nonlinearly.

\[
\boxed{
\text{attention = communication}
}
\]

\[
\boxed{
\text{FFN = computation at each token}
}
\]

---

### Slide 37 — Residual and normalization

Post-norm style:

\[
x'=\operatorname{LN}(x+\operatorname{MHA}(x))
\]

\[
y=\operatorname{LN}(x'+\operatorname{FFN}(x')).
\]

Pre-norm style:

\[
x'=x+\operatorname{MHA}(\operatorname{LN}(x))
\]

\[
y=x'+\operatorname{FFN}(\operatorname{LN}(x')).
\]

Pre-norm is often more stable for deep Transformers.

---

### Slide 38 — Transformer encoder block diagram

```text
x
│
├─ Multi-head self-attention
│
+ residual
│
LayerNorm
│
├─ Feed-forward network
│
+ residual
│
LayerNorm
```

Mention pre-norm/post-norm variant.

---

### Slide 39 — Transformer decoder block

Decoder block has:

1. causal self-attention;
2. cross-attention to encoder outputs;
3. feed-forward network.

Cross-attention uses:

\[
Q\text{ from decoder}
\]

\[
K,V\text{ from encoder}.
\]

Save detailed encoder-decoder Transformer for next lecture.

---

### Slide 40 — Decoder-only block

GPT-style language model:

1. causal self-attention;
2. FFN;
3. residuals and LayerNorm.

No encoder.

Output:

\[
p(x_{t+1}\mid x_{\le t})
=
\operatorname{softmax}(W_oh_t).
\]

---

### Slide 41 — Interactive 5: Transformer block

`transformer-block.html`

Show:

- token embeddings;
- positional encodings;
- attention matrix;
- residual stream;
- LayerNorm;
- FFN;
- output logits.

---

# Part VIII — Complexity and comparison

### Slide 42 — Self-attention cost

Attention matrix:

\[
A\in\mathbb R^{T\times T}.
\]

Compute:

\[
QK^\top
\]

costs approximately:

\[
O(T^2d).
\]

Memory:

\[
O(T^2).
\]

This is the main long-context bottleneck.

---

### Slide 43 — Compare path length

RNN path from token \(i\) to \(j\):

\[
O(|i-j|)
\]

CNN with dilation:

\[
O(\log |i-j|)
\]

Self-attention:

\[
O(1)
\]

in one layer, any token can attend to any other token.

---

### Slide 44 — Compare models

| Model | Parallel training | Long-range path | Cost |
|---|---:|---:|---:|
| RNN | low | long | \(O(Td^2)\) |
| TCN | high | medium | \(O(Tkd^2)\) |
| Transformer | high | short | \(O(T^2d)\) |

---

### Slide 45 — Why Transformers won many sequence tasks

They combine:

- parallel training;
- dynamic content-dependent context;
- short paths between tokens;
- scalable blocks;
- easy stacking with residuals/norm.

The original paper introduced the Transformer as an attention-only architecture and reported strong translation results with better parallelization than recurrent/convolutional alternatives.

---

# Part IX — Worked mini Transformer LM

### Slide 46 — Tiny decoder-only LM

Sequence:

```text
<BOS> I like
```

Goal:

```text
I like cats
```

Pipeline:

\[
\text{token IDs}
\rightarrow
\text{embeddings}
\rightarrow
+\text{positional encodings}
\rightarrow
\text{causal self-attention}
\rightarrow
\text{FFN}
\rightarrow
\text{logits}
\rightarrow
\text{CE loss}.
\]

---

### Slide 47 — Training loss

For input tokens:

\[
x_1,\ldots,x_T
\]

logits at position \(t\) predict:

\[
x_{t+1}.
\]

Loss:

\[
L=
-\sum_{t=1}^{T-1}
\log p_\theta(x_{t+1}\mid x_{\le t}).
\]

Same next-token objective as before.

---

### Slide 48 — Shifted targets

Input:

```text
<BOS> I like cats
```

Target:

```text
I     like cats <EOS>
```

Causal mask ensures position \(t\) cannot see target \(x_{t+1}\).

---

### Slide 49 — Interactive 6: tiny Transformer LM

`tiny-transformer-lm.html`

Show:

- token embeddings;
- causal mask;
- attention scores;
- attention weights;
- logits at each position;
- shifted targets;
- loss.

---

# Part X — Summary

### Slide 50 — What changed from RNNs?

RNN:

\[
h_t=f(h_{t-1},x_t)
\]

Transformer:

\[
H'=\operatorname{SelfAttention}(H)
\]

RNN carries memory through state.

Transformer computes interactions directly between positions.

---

### Slide 51 — Key equations

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+\text{mask}\right)
\]

\[
O=AV
\]

\[
\operatorname{MHA}(X)=[O_1;\ldots;O_H]W_O
\]

---

### Slide 52 — Final mental model

\[
\boxed{
\text{Self-attention = every token asks other tokens for information.}
}
\]

\[
\boxed{
\text{Multi-head = several relation types in parallel.}
}
\]

\[
\boxed{
\text{Positional encoding = order information.}
}
\]

\[
\boxed{
\text{Transformer block = attention + tokenwise MLP + residual/norm.}
}
\]

---

### Slide 53 — Next lecture

**Transformers II**

Topics:

- encoder-only models;
- decoder-only models;
- encoder-decoder Transformers;
- BERT;
- GPT;
- pretraining;
- finetuning;
- generation.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Motivation | 1–4 | 6 min |
| Attention as lookup | 5–10 | 10 min |
| Self-attention | 11–16 | 10 min |
| Causal masking | 17–21 | 8 min |
| Multi-head attention | 22–26 | 8 min |
| Positional encoding | 27–33 | 10 min |
| Transformer block | 34–41 | 12 min |
| Complexity/comparison | 42–45 | 7 min |
| Mini Transformer LM | 46–49 | 6 min |
| Summary | 50–53 | 3 min |

---

# Essential diagrams

1. RNN vs self-attention connectivity.
2. Q/K/V soft lookup.
3. \(QK^\top\) score matrix.
4. Attention heatmap.
5. Causal lower-triangular mask.
6. Multi-head attention block.
7. Positional encoding waves.
8. Transformer block residual stream.
9. Decoder-only next-token pipeline.
10. Complexity/path-length comparison.

---

# Interactives

1. `attention-lookup.html`
2. `self-attention-matrix.html`
3. `causal-attention.html`
4. `multi-head-attention.html`
5. `positional-encoding.html`
6. `transformer-block.html`
7. `tiny-transformer-lm.html`

---

# Notebooks

1. `01_scaled_dot_product_attention.ipynb`
2. `02_self_attention_from_scratch.ipynb`
3. `03_causal_masking.ipynb`
4. `04_multi_head_attention.ipynb`
5. `05_positional_encoding.ipynb`
6. `06_tiny_transformer_block.ipynb`
7. `07_character_transformer_lm.ipynb`
