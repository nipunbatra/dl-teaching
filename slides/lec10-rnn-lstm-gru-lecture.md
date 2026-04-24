---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# RNNs, LSTMs &amp; GRUs

## Lecture 10 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain why an **MLP** fails on variable-length sequences.
2. Write a **vanilla RNN** cell in 4 lines of PyTorch.
3. Diagnose the **vanishing/exploding** gradient in **BPTT**.
4. Explain how **LSTM gates** sidestep vanishing gradients.
5. Contrast **LSTM vs GRU** and know when each is appropriate.
6. Name 2026 niches where RNNs (or Mamba/RWKV) still win.

---

# Where we are

Module 5 opens. Until now everything was **feed-forward** — MLP, CNN. One input goes in, one output comes out, no memory.

Many real problems aren't like that:
- "He scored." — ambiguous without context.
- Audio, video, time series — ordered data.
- Language — tokens conditioned on all previous tokens.

<div class="paper">

Today maps to **Bishop Ch 12** (RNNs). UDL skips RNNs and jumps to Transformers — we cover the recurrent ideas first because they motivate attention (L12).

</div>

---

# Four questions

1. Why can't we just use an MLP for sequences?
2. What does a vanilla RNN actually compute?
3. Why do RNNs struggle with long-range dependencies — and how do LSTMs fix it?
4. When should you still use RNNs in 2026?

---

<!-- _class: section-divider -->

### PART 1

# Why MLPs fail on sequences

The parameter-sharing argument

---

# The problem

Suppose you want to classify "He lives in Gandhinagar." as positive sentiment.

If you feed this to an MLP:
- Words have no inherent order — must one-hot the position: `(token, position)`.
- Vocabulary is 50k words × max length 100 → input dim = 5,000,000.
- No reuse: the word "Gandhinagar" at position 5 is a completely different feature from the same word at position 23.

<div class="keypoint">

An MLP has **no inductive bias for time** — it would need to relearn what each word means, once per position.

</div>

---

# Weight sharing across time · the RNN trick

**Key idea**: the same network processes each timestep, with a *memory* that carries across.

![w:920px](figures/lec10/svg/rnn_unrolled.svg)

---

# RNN · step-by-step on "I love deep learning"

Embed each token to a 4-d vector, $x_1, x_2, x_3, x_4$. Initialize $h_0 = \mathbf{0}$.

<div class="math-box">

| step | input | hidden update | result |
|:-:|:-:|:-:|:-:|
| 1 | $x_1$ (I) | $\tanh(W x_1 + U h_0)$ | $h_1$ · encodes "I" |
| 2 | $x_2$ (love) | $\tanh(W x_2 + U h_1)$ | $h_2$ · encodes "I love" |
| 3 | $x_3$ (deep) | $\tanh(W x_3 + U h_2)$ | $h_3$ · encodes "I love deep" |
| 4 | $x_4$ (learning) | $\tanh(W x_4 + U h_3)$ | $h_4$ · full sentence summary |

</div>

$W, U$ are **shared** across steps. The same weights process "I" at step 1 and "learning" at step 4 — unlike an MLP, which would learn a separate `(token, position)` feature at every slot.

---

# RNN I/O patterns · three shapes

![w:920px](figures/lec10/svg/rnn_three_patterns.svg)

---

# The three RNN use-patterns

<div class="columns">
<div>

### Many-to-one

- Input: sequence
- Output: one label at the end
- e.g., sentiment classification

</div>
<div>

### One-to-many

- Input: one feature vector
- Output: a sequence
- e.g., image captioning

</div>
</div>

**Many-to-many** (in sync) · e.g., POS tagging — one label per input token.
**Many-to-many** (encoder-decoder) · e.g., translation — input sequence, output sequence, different lengths. This is the Seq2Seq pattern covered in L11.

Four shapes, same cell.

---

# RNN in PyTorch · by hand

```python
class RNNCell(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.W = nn.Linear(d_in, d_h, bias=False)
        self.U = nn.Linear(d_h,  d_h, bias=True)
        # tanh folds the output back to [-1, 1]

    def forward(self, x_t, h_prev):
        return torch.tanh(self.W(x_t) + self.U(h_prev))

# Unrolled loop
h = torch.zeros(batch, d_h)
for t in range(seq_len):
    h = cell(x[:, t], h)
```

`nn.RNN`, `nn.LSTM`, `nn.GRU` handle the loop + CUDA kernels for you.

---

<!-- _class: section-divider -->

### PART 2

# BPTT and vanishing gradients in time

Same problem as depth, now along the time axis

---

# Backpropagation Through Time

Unrolling gives you an $L$-step computation graph. Backprop through it:

$$\frac{\partial \mathcal{L}}{\partial h_1} = \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}} \cdot \frac{\partial \mathcal{L}}{\partial h_T}$$

Each factor is $W^\top \cdot \text{diag}(\tanh'(\cdot))$.

**Same problem as deep networks** — product of many Jacobians. If the spectral radius of $W$ is $< 1$, gradient vanishes. If $> 1$, it explodes.

---

# Vanishing gradients in time

![w:920px](figures/lec10/svg/bptt_vanishing.svg)

---

# Worked example · BPTT on 3 timesteps

<div class="math-box">

Consider a tiny RNN with scalar state · $h_t = \tanh(w h_{t-1} + x_t)$ · $w = 0.5$ · $\tanh'(\cdot) \le 1$.

$$\frac{\partial h_3}{\partial h_0} = \underbrace{w \tanh'(\cdot)}_{\le 0.5} \cdot \underbrace{w \tanh'(\cdot)}_{\le 0.5} \cdot \underbrace{w \tanh'(\cdot)}_{\le 0.5} \approx 0.125$$

- 3 steps · gradient at most 0.125 (already 8× smaller)
- 10 steps · $\le 0.5^{10} \approx 10^{-3}$
- 50 steps · $\le 0.5^{50} \approx 10^{-15}$

</div>

**That's why vanilla RNNs can't learn dependencies across more than ~20 timesteps.** Every step in the product pulls the gradient toward zero if $|w \tanh'| < 1$, or toward infinity if $> 1$. LSTMs (next) sidestep this via additive gated updates.

---

# Truncated BPTT · the practical fix

For long sequences (thousands of steps), full BPTT is expensive.

**Truncated BPTT (TBPTT)** — only backpropagate $K$ steps at a time:

```python
for chunk in sequence.split(K, dim=1):
    h_detached = h.detach()             # cut gradient here
    for t in range(chunk.size(1)):
        h = cell(chunk[:, t], h)
    loss = criterion(h, target)
    loss.backward()
    opt.step()
```

Typical $K = 32$ to $256$. Keeps training tractable at the cost of losing very-long-range gradient signal.

---

# Gradient clipping · the second fix

Exploding gradients are worse than vanishing — one bad step can destroy weeks of training. Clip by global norm:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
```

<div class="keypoint">

Pascanu et al. 2013 showed clipping at norm ~1 makes RNN training robust. Still the default for any sequence model — RNN, LSTM, Transformer. Cheap insurance against numerical catastrophe.

</div>

---

<!-- _class: section-divider -->

### PART 3

# LSTM · the gating fix

Three sigmoid gates protect a cell state

---

# LSTM cell · annotated

![w:920px](figures/lec10/svg/lstm_annotated.svg)

---

# LSTM cell architecture

![w:920px](figures/lec10/svg/lstm_cell.svg)

<div class="realworld">

▶ Interactive: drag forget/input/output sliders; see the cell state freeze, flow, or reset — [lstm-gates](https://nipunbatra.github.io/interactive-articles/lstm-gates/).

</div>

---

# LSTM · the six equations

<div class="math-box">

$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_f) \quad \text{(forget gate)}$$

$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_i) \quad \text{(input gate)}$$

$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_o) \quad \text{(output gate)}$$

$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_c) \quad \text{(candidate)}$$

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \quad \text{(cell update)}$$

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t) \quad \text{(hidden output)}$$

</div>

Four learnable weight matrices · three gates in $[0, 1]$ · one candidate in $[-1, 1]$.

---

# Each gate · in plain English

- **Forget gate $\mathbf{f}_t$** · "what should I erase from the cell state?"
  - Close to 0 · forget (reset a counter, flush old context)
  - Close to 1 · keep it around (persistent memory)

- **Input gate $\mathbf{i}_t$** · "how much of the new candidate should I actually write?"
  - Close to 0 · ignore this input
  - Close to 1 · accept fully

- **Output gate $\mathbf{o}_t$** · "what of the cell state do I expose to downstream layers?"
  - Close to 0 · keep memory silent
  - Close to 1 · project it out

<div class="keypoint">

The LSTM's "memory" is the cell state $\mathbf{c}_t$; the gates are **learned controllers** that decide when to write, keep, or read. Think of it as a differentiable tiny memory cell plus a learned read/write scheduler.

</div>

---

# Why gating fixes vanishing gradients

The cell state update is **additive**:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

If $\mathbf{f}_t = \mathbf{1}$ and $\mathbf{i}_t = \mathbf{0}$, then $\mathbf{c}_t = \mathbf{c}_{t-1}$ — memory is **preserved perfectly**.

<div class="keypoint">

Gradient path along the cell state is a **multiply-by-$\mathbf{f}_t$** only. No matrix multiply, no tanh Jacobian. With $\mathbf{f}_t \approx 1$, the gradient flows unchanged across hundreds of steps.

</div>

This is the same idea as ResNet skip connections, in the time dimension.

---

<!-- _class: section-divider -->

### PART 4

# GRU · the lighter sibling

Fewer gates, comparable accuracy

---

# GRU · two gates instead of three

Cho et al. 2014 merged LSTM's forget and input gates into a single **update gate** $\mathbf{z}$, and dropped the separate cell state.

<div class="math-box">

$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(update gate)}$$

$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t]) \quad \text{(reset gate)}$$

$$\tilde{\mathbf{h}}_t = \tanh(W [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

</div>

---

# LSTM vs GRU · a history sentence

In the late 2010s, ML papers often included an ablation · "we tried LSTM and GRU and picked whichever worked." By 2019, most groups defaulted to whichever they had better library support for.

<div class="insight">

That casualness told the story · **the gating trick matters; which gates you pick doesn't much.** Any additive-gated recurrence works; the architectural variants are micro-optimizations on the core idea.

</div>

---

# LSTM vs GRU · when to pick which

| | LSTM | GRU |
|---|------|-----|
| Gates | 3 + candidate | 2 + candidate |
| State | cell + hidden | hidden only |
| Params (d_h = 128) | 4 · 128 · 256 = 131k | 3 · 128 · 256 = 98k |
| Accuracy | baseline | often tied |
| Training speed | slower | ~15% faster |

<div class="insight">

Empirically close — both far beyond vanilla RNNs on long-range tasks. GRU was often preferred pre-Transformer; today either is fine for the few remaining RNN use cases.

</div>

---

# Bidirectional + stacked RNNs

**Bidirectional** · run one RNN left-to-right and another right-to-left; concatenate the outputs. Used for classification/tagging (not generation).

```python
self.rnn = nn.LSTM(d_in, d_h, num_layers=2, bidirectional=True, batch_first=True)
```

**Stacked** (deep) RNN · layer $\ell$'s hidden state feeds layer $\ell+1$. Each layer captures different abstraction.

Both tricks combinable: 2-layer bidirectional LSTMs were the standard NLP architecture from 2015–2017 (before Transformers).

---

<!-- _class: section-divider -->

### PART 5

# When to still use RNNs in 2026

---

# The 2026 reality

Transformers have largely replaced RNNs for:
- **Language modeling** (GPT, Llama, Claude)
- **Machine translation** (Google Translate, DeepL)
- **Speech recognition** (Whisper)
- **Music generation** (MusicLM)

But RNNs are still the right choice for:

<div class="columns">
<div>

**Streaming / online inference**
- Process input one token at a time
- O(1) state update
- No need to recompute attention over history

</div>
<div>

**Tiny devices**
- Microcontrollers, always-on sensors
- Memory budget in KB
- LSTM cell: ~1k params

</div>
</div>

---

# RWKV and Mamba · the RNN comeback

Starting in 2023, a new class of models has re-emerged · **state-space models** and **linear RNNs** that match Transformer quality with **O(1) inference per token**.

<div class="math-box">

- **RWKV** (Peng 2023) · linear attention reformulated as RNN · trained in parallel, runs as recurrent at inference.
- **Mamba** (Gu 2023) · selective state-space model · same asymptotic scaling, competitive on language.
- **Mamba-2** (2024) · faster, matches Transformer-7B quality.

</div>

<div class="insight">

The story isn't "RNNs are dead" — it's "vanilla RNNs with sequential gradients couldn't scale." Modern parallelizable RNNs are a quiet comeback. Watch this space.

</div>

---

# A preview · the problem RNNs can't solve

Consider translating:

> "The animal didn't cross the street because it was too tired."

To translate "it" correctly, the model must look back to "animal" — maybe 6 tokens ago.

An RNN compresses all of that into a single $h_t$ vector. Longer sentences → more to compress → more is lost.

<div class="keypoint">

The next lecture (L11) examines encoder-decoder Seq2Seq, which also struggles with this **fixed-length bottleneck**. That struggle motivates **attention** (L12) — the idea that finally let sequence models scale.

</div>

---

<!-- _class: summary-slide -->

# Lecture 10 — summary

- **MLPs fail on sequences** because they can't share parameters across time.
- **RNN** — same cell, shared weights, hidden state carries memory forward.
- **BPTT** — backprop through unrolled graph; same vanishing/exploding problem as depth.
- **LSTM** — gated cell state is an additive "conveyor belt"; gradients flow through gates, not through tanh products.
- **GRU** — simpler (2 gates instead of 3); often equivalent accuracy, ~15% faster.
- **2026** — Transformers own most sequence tasks, but RNNs still win for streaming and tiny devices.

### Read before Lecture 11

Bishop Ch 12 · Seq2Seq.

### Next lecture

**Seq2Seq + the motivation for attention** — encoder-decoder architecture, teacher forcing, beam search, and the bottleneck that made attention inevitable.

<div class="notebook">

**Notebook 10** · `10-lstm-from-scratch.ipynb` — implement an LSTMCell with only `nn.Linear` layers; verify output matches `nn.LSTMCell`; train a char-level LSTM on Tiny Shakespeare.

</div>
