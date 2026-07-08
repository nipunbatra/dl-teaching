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
6. Name the niches where RNNs (or Mamba/RWKV) still win today.

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

# Bridge from ES 335 · you already met the RNN

| From ES 335 | What we do today |
|---|---|
| The recurrence $h_t = \tanh(W_{hh}\,h_{t-1} + W_{xh}\,x_t + b_h)$ | **unroll** it in time and run BPTT through every step |
| *Conceptually* · repeated $W_{hh}$ makes signals **explode** ($\|W_{hh}\|>1$) or **vanish** ($\|W_{hh}\|<1$) | **derive** that exponential blow-up / decay from the chain rule |
| You heard the LSTM has **3 gates** (forget, input, output) | write the **gate math** and show *why* it rescues the gradient |

<div class="insight">

In ES 335 you saw simple RNNs vanish/explode via repeated $W_{hh}$ — as a picture. Today we unroll the net, **derive** exactly how BPTT fails, then **derive the LSTM gate math** that fixes it. Problem → failure → fix.

</div>

---

# Four questions

1. Why can't we just use an MLP for sequences?
2. What does a vanilla RNN actually compute?
3. Why do RNNs struggle with long-range dependencies — and how do LSTMs fix it?
4. When should you still use RNNs today?

---

# Pop quiz · finish the sentence

> *"After the long flight from Mumbai, my parents finally arrived at the Frankfurt airport, and **they** ____."*

<div class="popquiz">

Whose pronoun is "they" referring to? Mumbai? The flight? The parents?

(a) An MLP given the sentence as a flat 100-dim bag-of-words vector.
(b) An MLP given the **last 5 words only**.
(c) A model with **memory** that can carry "parents" forward through the sentence.

Stop and think about (a) and (b). Why do they fail? That failure is the entire reason RNNs exist.

</div>

---

▶ **Interactive** · click LSTM gates to see them open and close on a real sentence → [lstm-gates](https://nipunbatra.github.io/interactive-articles/lstm-gates/) · explore vanishing gradients → [vanishing-gradients](https://nipunbatra.github.io/interactive-articles/vanishing-gradients/).

---

<!-- _class: section-divider -->

### PART 1

# Why MLPs fail on sequences

Why can't an MLP just read the whole sentence?

---

# The problem

Suppose you want to classify "He lives in Gandhinagar." as positive sentiment.

If you feed this to an MLP:
- **Length varies.** A tweet is 5 words, a review is 500 — but an MLP's input is a *fixed-size* vector. What size do you pick?
- Words have no inherent order — must one-hot the position: `(token, position)`.
- Vocabulary is 50k words × max length 100 → input dim = 5,000,000.
- No reuse: the word "Gandhinagar" at position 5 is a completely different feature from the same word at position 23.

<div class="keypoint">

An MLP has **no inductive bias for time** — it would need to relearn what each word means, once per position.

</div>

---

# First, the object · what does one RNN cell output?

Before any equation, fix the **one thing** an RNN produces: a cell reads the **new word** $x_t$ and the **old memory** $h_{t-1}$, and returns a **new memory** $h_t$ — a small vector summarizing the sequence *so far*.

![w:760px](figures/lec10/svg/rnn_cell_output.svg)

<div class="keypoint">

**Big idea (2 sentences).** An RNN cell is a memory-updater · *new state = f(old state, new word)*. Everything else today — unrolling, BPTT, gates — just computes and *protects* that one running summary $h_t$.

</div>

---

# Weight sharing across time · the RNN trick

**Key idea**: the same network processes each timestep, with a *memory* that carries across.

![w:920px](figures/lec10/svg/rnn_unrolled.svg)

---

# RNN · step-by-step on "I love deep learning"

Tiny model · input dim $d_\text{in} = 2$, hidden dim $d_h = 3$. $h_0 = [0, 0, 0]^\top$. Embeddings (toy):
$x_1 = [1, 0]^\top$ (I), $x_2 = [0, 1]^\top$ (love), $x_3 = [1, 1]^\top$ (deep), $x_4 = [0, 0]^\top$ (learning).

Weights ($W$ is the input matrix $W_{xh}$, $U$ is the recurrent matrix $W_{hh}$ from the bridge slide):
$$W = \begin{pmatrix} 0.1 & 0.2 \\ 0.3 & 0.4 \\ 0.5 & 0.6 \end{pmatrix},\quad U = \begin{pmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ 0.7 & 0.8 & 0.9 \end{pmatrix}$$

**Step 1 · "I".** $h_0 = 0$, so $h_1 = \tanh(Wx_1) = \tanh([0.1, 0.3, 0.5]^\top) \approx [0.099, 0.291, 0.462]^\top$. This vector summarizes the sequence so far.

**Step 2 · "love".** Use the **same** $W, U$ with new input $x_2$ and previous state $h_1$:
$$h_2 = \tanh(\underbrace{Wx_2}_{[0.2,\,0.4,\,0.6]^\top} + \underbrace{Uh_1}_{[0.21,\,0.46,\,0.72]^\top}) = \tanh([0.41,\,0.86,\,1.32]^\top) \approx [0.39,\,0.70,\,0.87]^\top$$

Same weights, but now the state carries information from **both** words. That reuse — one $W, U$ for every step — is the whole RNN idea.

---

# RNN · the recurrence at a glance

Continue the same pattern for all four words:
| step | input | update | result |
|:-:|:-:|:-:|:-:|
| 1 | $x_1$ | $\tanh(Wx_1 + Uh_0)$ | $h_1$ encodes "I" |
| 2 | $x_2$ | $\tanh(Wx_2 + Uh_1)$ | $h_2$ encodes "I love" |
| 3 | $x_3$ | $\tanh(Wx_3 + Uh_2)$ | $h_3$ encodes "I love deep" |
| 4 | $x_4$ | $\tanh(Wx_4 + Uh_3)$ | $h_4$ — full summary |

$W, U$ are **shared** across steps — unlike an MLP that would learn a different `(token, position)` feature for every slot.

---

# RNN I/O patterns · four shapes

![w:920px](figures/lec10/svg/rnn_three_patterns.svg)

---

# The four RNN use-patterns

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

What happens to a gradient that travels back 50 steps?

---

# BPTT · the telephone game

<div class="insight">

**Analogy · Chinese whispers.** Tell a secret to person 1. They whisper to person 2, etc. By person 20, the message is garbled (vanished) or wildly distorted (exploded).

The gradient is that secret. It travels backward from the loss to the start of the sequence, and at each step it gets transformed.

</div>

---

# ⭐⭐⭐ Optional · bptt · derive the gradient product

**Gist ·** the gradient back through $T$ steps is a product of $T$ copies of (roughly) $W_{hh}$ — so it shrinks or blows up **exponentially** in $T$. The algebra below just makes that precise.

Simplest possible RNN: $h_t = W_{hh}\, h_{t-1}$ (no input, no $\tanh$), so $h_t = W_{hh}^{\,t}\, h_0$. Chain rule over 3 steps:
$$\frac{\partial h_3}{\partial h_0} = \underbrace{\frac{\partial h_3}{\partial h_2}}_{W_{hh}} \cdot \underbrace{\frac{\partial h_2}{\partial h_1}}_{W_{hh}} \cdot \underbrace{\frac{\partial h_1}{\partial h_0}}_{W_{hh}} = W_{hh}^{\,3} \;\Longrightarrow\; \frac{\partial h_T}{\partial h_0} = W_{hh}^{\,T}$$

Add $\tanh$ back ($h_t = \tanh(W_{hh} h_{t-1})$): each Jacobian becomes $W_{hh}^\top \cdot \text{diag}(\tanh'(\cdot))$ with $\tanh' \in (0,1)$, so
$$\frac{\partial \mathcal{L}}{\partial h_1} = \frac{\partial \mathcal{L}}{\partial h_T} \cdot \prod_{t=2}^T \frac{\partial h_t}{\partial h_{t-1}} \;=\; \text{product of } T \text{ scaled-down matrices.}$$

Vanishing if $\|W_{hh}\| < 1$, exploding if $\|W_{hh}\| > 1$ — exponential in $T$ either way.

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

**That's why vanilla RNNs typically struggle to learn dependencies beyond a few tens of timesteps.** Every step in the product pulls the gradient toward zero if $|w \tanh'| < 1$, or toward infinity if $> 1$. LSTMs (next) sidestep this via additive gated updates.

---

# Truncated BPTT · the practical fix

For long sequences (thousands of steps), full BPTT is expensive.

**Truncated BPTT (TBPTT)** — only backpropagate $K$ steps at a time:

```python
for chunk in sequence.split(K, dim=1):
    h = h.detach()                      # cut gradient here — truncate BPTT
    for t in range(chunk.size(1)):
        h = cell(chunk[:, t], h)
    loss = criterion(h, target)
    loss.backward()
    opt.step()
```

Typical $K = 32$ to $256$. Keeps training tractable at the cost of losing very-long-range gradient signal.

<div class="keypoint">

Why bother? BPTT must **store every intermediate $h_t$** for the backward pass — activation memory grows $O(T)$ with sequence length. Truncation caps both compute *and* memory at $O(K)$.

</div>

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

How do you keep one memory alive for 100 steps?

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

# LSTM · the conveyor-belt analogy

<div class="insight">

A vanilla RNN crams everything into one memory $h_t$ — like doing complex calculations using only 8 CPU registers; things get overwritten constantly.

LSTM adds a separate **memory conveyor belt** · the **cell state $\mathbf{c}_t$**. Information travels along this belt unchanged unless special **gates** intervene:
- **Forget gate** · the janitor — removes items from the belt.
- **Input gate** · the gatekeeper — decides what new items to add.
- **Output gate** · the press secretary — picks what to show outside.

</div>

A protected long-term memory + learned controllers for write / forget / read. Keep the three roles in mind — next we ask what each gate actually *outputs*.

---

# First, the object · what does each gate output?

Each gate outputs a **0–1 mask** — one number per memory slot — that decides *keep / write / read*. Same recipe for all three · squash $[\mathbf{h}_{t-1}, \mathbf{x}_t]$ through a sigmoid, so every value lands in $[0,1]$.

- **Forget gate $\mathbf{f}_t$** · *"what do I erase?"* — 0 = flush old context · 1 = keep it.
- **Input gate $\mathbf{i}_t$** · *"how much new info do I write?"* — 0 = ignore · 1 = accept fully.
- **Output gate $\mathbf{o}_t$** · *"what do I expose downstream?"* — 0 = stay silent · 1 = project out.

<div class="keypoint">

**Big idea (2 sentences).** The memory is the cell state $\mathbf{c}_t$; the three gates are learned dials in $[0,1]$ that schedule when to write, keep, or read it. Get the *masks* first — the equations below are just "sigmoid gives the mask, the mask multiplies the memory."

</div>

---

# LSTM · the update that saves gradients

Given the three masks and a **candidate** $\tilde{\mathbf{c}}_t$ ($\tanh$, so in $[-1,1]$) — the new information on offer:

**Update the cell state** · keep some old, add some new:
$$\mathbf{c}_t = \underbrace{\mathbf{f}_t \odot \mathbf{c}_{t-1}}_{\text{kept}} + \underbrace{\mathbf{i}_t \odot \tilde{\mathbf{c}}_t}_{\text{added}}$$

**Read out** the hidden state through the output mask:
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

<div class="keypoint">

The gates are just knobs. The crucial thing is the **+** in the cell-state update: memory is *added to*, not matrix-multiplied through — and that single **+** is what keeps gradients alive over long sequences (we prove it two slides on).

</div>

---

# ⭐⭐⭐ Optional · LSTM · the gate algebra

**Gist ·** every gate is one sigmoid on $[\mathbf{h}_{t-1}, \mathbf{x}_t]$; the candidate is one $\tanh$. Nothing more.

Stack the previous hidden state and current input into one control vector $[\mathbf{h}_{t-1}, \mathbf{x}_t]$, then:
$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_f),\qquad \mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_i)$$
$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_o),\qquad \tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}, \mathbf{x}_t] + b_c)$$

Four little linear layers, four nonlinearities. Three sigmoids make masks in $[0,1]$; one $\tanh$ makes the candidate in $[-1,1]$. That is the entire LSTM parameter set.

---

# Worked numeric · single-neuron LSTM

1D state. Setup:
- $c_{t-1} = 5.0$ ("the subject is plural")
- $h_{t-1} = 0.9$, new input $x_t = 0.2$ (the word "was")

Network has learned (for this input):
- **Forget** $f_t = 0.1$ — sees "was" → forget the plural memory.
- **Input** $i_t = 0.9$ — write new info aggressively.
- **Candidate** $\tilde c_t = -1.0$ — a $\tanh$ output (so in $[-1,1]$), encoding "subject is singular".

**Compute new cell state:**
$$c_t = f_t \cdot c_{t-1} + i_t \cdot \tilde c_t = 0.1 \cdot 5.0 + 0.9 \cdot (-1.0) = 0.5 - 0.9 = \mathbf{-0.4}$$

The memory **flipped** from large positive (plural) to negative (singular) in one step — exactly because the forget gate was small *and* the input gate was large.

---

# The gradient highway · picture first

![w:860px](figures/lec10/svg/additive_vs_multiplicative.svg)

<div class="insight">

The vanilla path **multiplies** the gradient by a matrix at every hop → it shrinks (or explodes) exponentially. The LSTM cell state is **added** to → the gradient rides a clean highway, scaled only by $f_t \approx 1$. Same trick as a ResNet skip, but across *time* instead of depth.

</div>

---

# Why gating fixes vanishing gradients

**Vanilla RNN.** $h_t = \tanh(Wh_{t-1} + \cdots)$. Backward Jacobian:
$$\frac{\partial h_t}{\partial h_{t-1}} = W^\top \cdot \text{diag}(\tanh'(\cdot))$$
A **matrix multiplication** at every step → product of matrices → vanishing/exploding.

**LSTM cell state.** $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$. Along the cell-state path:
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

That's it — **no matrix multiplication**. Just element-wise multiplication by the forget gate.

If the network learns $f_t \approx 1$ ("don't forget"), the gradient flows through unchanged. The "gradient highway" works in time the same way ResNet's identity skip works in depth.

---

# Worked numeric · gradient flow over 100 steps

Memory must survive → forget gate trained to $f_t = 0.99$. Vanilla RNN (optimistic factor $0.5$) for contrast:

![w:660px](figures/lec10/svg/gradient_survival.svg)

$0.99^{100} \approx \mathbf{0.366}$ (usable) vs $0.5^{100} \approx 7.9\times10^{-31}$ (below precision). LSTM signal **survives** — same idea as a ResNet skip, in the time dimension.

---

# Now it clicks · the whole arc in one line

<div class="insight">

**Problem → failure → fix.**
MLPs can't share across time → an **RNN** shares one cell and carries a running memory $h_t$ → but **BPTT** multiplies that memory by $W$ at every step, so gradients **vanish** → the **LSTM** swaps the multiplicative path for an **additive** cell state guarded by 0–1 gates → long-range memory **survives**.

</div>

Every piece we built — unroll, BPTT, the gate masks, the **+** in $\mathbf{c}_t$ — exists to serve that one arc. If the arc makes sense, the equations are just bookkeeping.

---

<!-- _class: section-divider -->

### PART 4

# GRU · the lighter sibling

Do we really need all three gates?

---

# GRU · a simpler, smarter gate

Two design questions Cho et al. asked in 2014:
1. Do we really need a *private* cell state $c_t$ and a *public* hidden state $h_t$? Just keep one.
2. The LSTM's forget and input gates are often opposites. Can we use one gate that picks "old vs new"?

Result · the **update gate** $\mathbf{z}_t$ is a dial:
- $\mathbf{z}_t \approx 0$ → keep the old state.
- $\mathbf{z}_t \approx 1$ → switch to the new candidate.

---

# GRU · build the equations step-by-step

**Step 1 · Reset gate.** How much of the past to use when forming the candidate.
$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Step 2 · Candidate.** Reset gate filters the past first.
$$\tilde{\mathbf{h}}_t = \tanh(W [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Step 3 · Update gate.** How much new vs old to use.
$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}, \mathbf{x}_t])$$

**Step 4 · Final state · linear interpolation.**
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

The **additive** structure (like LSTM's cell state) is what keeps gradients flowing.

*(Convention note · PyTorch's `nn.GRU` swaps the roles, $\mathbf{h}_t = \mathbf{z}_t \odot \mathbf{h}_{t-1} + (1-\mathbf{z}_t) \odot \tilde{\mathbf{h}}_t$ — same model, relabeled gate.)*

---

# Worked numeric · scalar GRU

1D state. $h_{t-1} = 0.8$ ("expecting a verb"). Candidate $\tilde h_t = -0.5$ ("new word is a noun").

**Case 1 · $z_t = 0.1$ (keep old).**
$h_t = 0.9 \cdot 0.8 + 0.1 \cdot (-0.5) = 0.72 - 0.05 = \mathbf{0.67}$
Close to old state — input mostly ignored.

**Case 2 · $z_t = 0.9$ (use new).**
$h_t = 0.1 \cdot 0.8 + 0.9 \cdot (-0.5) = 0.08 - 0.45 = \mathbf{-0.37}$
Close to candidate — state nearly fully replaced.

The single update gate gives the network smooth control between "preserve" and "overwrite."

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

# When to still use RNNs today

Didn't Transformers kill them?

---

# The reality · as of this writing

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

# ⭐⭐⭐ Optional · rwkv and mamba · the rnn comeback

Starting in 2023, a new class of models has re-emerged · **state-space models** and **linear RNNs** that match Transformer quality with **O(1) inference per token**.

<div class="math-box">

- **RWKV** (Peng 2023) · linear attention reformulated as RNN · trained in parallel, runs as recurrent at inference.
- **Mamba** (Gu 2023) · selective state-space model · same asymptotic scaling, competitive on language.
- **Mamba-2** (2024) · faster, matches Transformer-7B quality.

The core of an SSM is a **linear recurrence** · $\mathbf{h}_t = \bar{A}\,\mathbf{h}_{t-1} + \bar{B}\,\mathbf{x}_t$, $\;\mathbf{y}_t = C\,\mathbf{h}_t$ — no $\tanh$ between steps, so the whole sequence unrolls into a parallel scan at training time.

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

# Putting it all together · the L10 master sentence

<div class="math-box">

A **recurrent net is just an MLP that re-uses the same weights at every step** and carries a state forward. Vanilla RNN does this naively · LSTM/GRU add **gates** that decide what to keep, forget, or write — and the gates are why long-range dependencies survive.

</div>

| Idea | What it is | Friendly intuition |
|:-:|:-:|:-:|
| Hidden state $h_t$ | a vector summarizing the past | "what I remember so far" |
| Recurrence | $h_t = f(h_{t-1}, x_t)$ | "update memory with new word" |
| Gating (LSTM) | learned switches on the cell | "should I forget, write, or read?" |
| BPTT | chain rule unrolled through time | "backprop, but $T$ steps deep" |

Same NLL story as L00 · we model $p(y_t \mid y_{<t}, x; \theta)$ · the only new thing is *how* we compute the conditioning vector.

---

# Pop quiz · revisit

The "they" pronoun? Only the model with memory (option **c**) can carry "my parents" forward across 14 words.

<div class="keypoint">

(a) Bag-of-words throws away order — *which* noun came right before "they" is gone.
(b) Last-5-words misses "parents" entirely.
(c) ✓ A recurrent state can hold "parents" the whole way.

</div>

The same need shows up in stock prices, audio, video. Anything ordered.

---

# Practice problems

<div class="math-box">

**P1.** A vanilla RNN has $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)$. With $h \in \mathbb{R}^{128}$ and $x \in \mathbb{R}^{50}$, count parameters. Compare to a feedforward net with $T = 100$ time-steps that *doesn't* share weights — how many?

**P2.** Show that the gradient $\partial h_T / \partial h_0$ is a product of $T$ Jacobians, each $W_{hh}^\top \,\text{diag}(\tanh')$. Argue why the spectral norm of this product can vanish (or explode) exponentially in $T$.

**P3.** An LSTM forget gate outputs $f_t = \sigma(W_f [h_{t-1}; x_t] + b_f)$. Initial bias $b_f = 1$ is standard practice. Why? (Hint · think $\sigma(1) \approx 0.73$.)

</div>

---

# Practice problems · gates &amp; usage

<div class="math-box">

**P4.** A GRU has fewer gates than an LSTM. Write the GRU update equations from memory and identify which two LSTM gates are merged.

**P5** *(L11 preview).* **Teacher forcing** at training, autoregressive at inference. Sketch one failure mode that arises when you switch (the *exposure bias* problem).

**P6.** When *would* you still pick an RNN over a Transformer today? Name two scenarios and the reason for each (hint · streaming, very long context with constant memory).

</div>

---

# What breaks · symptom → suspect → test

| Symptom | Suspect | Fastest test |
|---|---|---|
| Loss explodes to NaN after ~50 steps | exploding gradients through time | clip grad norm at 1.0, plot grad norms per step |
| Model nails recent tokens, blind beyond ~20 steps | vanishing gradients in the vanilla RNN | plot $\lVert\partial\mathcal{L}/\partial h_t\rVert$ vs $t$ · swap in LSTM/GRU |
| LSTM forgets everything early in training | forget gate starts near 0 | init $b_f = 1$ · log mean $f_t$ over a batch |
| Great teacher-forced loss, gibberish when sampling | exposure bias — train/inference mismatch | decode autoregressively on *training* sentences |
| Loss plateaus on long-range tasks despite LSTM | TBPTT window $K$ shorter than the dependency | increase $K$ · check where `h.detach()` cuts the graph |

---

<!-- _class: summary-slide -->

# Lecture 10 — summary

- **MLPs fail on sequences** because they can't share parameters across time.
- **RNN** — same cell, shared weights, hidden state carries memory forward.
- **BPTT** — backprop through unrolled graph; same vanishing/exploding problem as depth.
- **LSTM** — gated cell state is an additive "conveyor belt"; gradients flow through gates, not through tanh products.
- **GRU** — simpler (2 gates instead of 3); often equivalent accuracy, ~15% faster.
- **Currently** — Transformers own most sequence tasks, but RNNs still win for streaming and tiny devices.

### Read before Lecture 11

Bishop Ch 12 · Seq2Seq.

### Next lecture

**Seq2Seq + the motivation for attention** — encoder-decoder architecture, teacher forcing, beam search, and the bottleneck that made attention inevitable.

<div class="notebook">

**Notebook 10** · `10-lstm-from-scratch.ipynb` — implement an LSTMCell with only `nn.Linear` layers; verify output matches `nn.LSTMCell`; train a char-level LSTM on Tiny Shakespeare.

</div>

---

# The one-sentence takeaway

<div class="insight">

**LSTMs don't remember harder — they learn what to forget.**

</div>

*Next: two RNNs back to back, and the fixed-length bottleneck that strangles them.*
