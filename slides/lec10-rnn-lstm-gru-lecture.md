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

<!-- _class: section-divider -->

### PART 3

# LSTM · the gating fix

Three sigmoid gates protect a cell state

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
