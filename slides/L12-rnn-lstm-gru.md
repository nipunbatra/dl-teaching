---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Sequence Models I — RNNs, LSTMs & GRUs

## Lecture 12 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

An MLP sees a sentence as a fixed bag of numbers. A **recurrent net** reads it one token at a time, carrying a **running summary** $h_t$ forward — the *same* small network reused at every step.

<div class="keypoint">

**One cell, shared across time, plus a memory that flows.**
That reuse is the entire RNN. But training it — backprop through an unrolled net — multiplies the gradient by the *same matrix* at every step, so it **vanishes or explodes**. The **LSTM** fixes this with an *additive* memory that gradients ride like a highway.

</div>

You met IID mini-batches in L01 — and the one place IID *breaks* (stock prices, video, language). This is that lecture.

$$\text{weight sharing} \rightarrow \underbrace{\text{BPTT}}_{\text{vanish / explode}} \rightarrow \underbrace{\text{LSTM}}_{\text{additive } c_t} \rightarrow \text{GRU} \rightarrow \text{when RNNs still win}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Where today's ideas go next:**
- **L13** — two RNNs back-to-back (seq2seq); the fixed-length **bottleneck**
- **L14** — **attention** removes that bottleneck
- **L15+** — the Transformer drops recurrence entirely
- **Streaming / edge** — the RNN's $O(1)$ update still wins

</div>
<div>

**Borrowed from your own courses** (build on, not repeat):
- the recurrence $h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b)$ — **ML (ES 335)**
- weight sharing, vanish/explode *as a picture* — **ML (ES 335)**
- MLE / NLL — **L01** (an RNN just models $p(y_t\mid y_{<t})$)

</div>
</div>

<div class="insight">

ES 335 showed you the RNN cell and *said* gradients vanish. Today we **unroll**, **derive** why, and **derive the LSTM gate math** that rescues it. Problem → failure → fix.

</div>

---

<!-- _class: section-divider -->

## Part 1 · One cell, shared across time

---

# Why an MLP can't read a sequence

> *"After the long flight from Mumbai, my parents finally arrived — and **they** ____."*

<div class="popquiz">

**Warm-up.** Which model can resolve "they" = *parents*? (a) an MLP fed the sentence as a fixed 100-dim bag-of-words · (b) an MLP fed only the **last 5 words** · (c) a model with **memory** that carried "parents" forward.

</div>

An MLP wants a **fixed-size** input — but sentences vary from 5 to 500 words. Worse, it has **no inductive bias for time**: "parents" at position 3 is a *different* input feature from "parents" at position 30, so it must relearn every word once per slot.

*Answer:* only **(c)** — (a) throws away order, (b) never sees "parents." That failure is the whole reason RNNs exist.

---

# What one RNN cell outputs

Before any equation, fix the **one thing** a cell produces: it reads the **new token** $x_t$ and the **old memory** $h_{t-1}$, and returns a **new memory** $h_t$ — a vector summarizing the sequence *so far*.

![w:760px](figures/lec10/svg/rnn_cell_output.svg)

<div class="keypoint">

An RNN cell is a **memory-updater**: *new state = f(old state, new token)*. Everything else today — unrolling, BPTT, gates — just computes and *protects* that one running summary $h_t$.

</div>

---

# Intuition · an RNN is an MLP in a loop

Forget the word "recurrent" for a second. A vanilla RNN cell is **exactly one hidden layer of an MLP** — a matrix multiply and a $\tanh$. The only new idea: **wire its output back to its input.**

<div class="columns">
<div>

**MLP layer** — one shot:
$$h = \tanh(W_{xh}\,x + b)$$
in → hidden → out. Done.

</div>
<div>

**RNN cell** — the *same* layer, fed its own last output:
$$h_t = \tanh(W_{xh}\,x_t + \underbrace{W_{hh}\,h_{t-1}}_{\text{the loop}} + b)$$

</div>
</div>

<div class="keypoint">

**An RNN is an MLP in a `for` loop, carrying a memory $h_t$ forward.** The loop is the whole trick — it lets one fixed-size layer read a sequence of *any* length, because the memory $h_t$ is a running summary of everything seen so far.

</div>

---

# Weight sharing: the same net at every step

Unroll the recurrence in time and the picture is a chain — but every box is the **same** $W_{xh}, W_{hh}, b$:

$$h_t = \tanh(W_{xh}\,x_t + W_{hh}\,h_{t-1} + b), \qquad \hat y_t = W_{hy}\,h_t$$

![w:640px](figures/lec10/svg/rnn_unrolled.svg)

<div class="insight">

**One parameter set, unlimited length.** An MLP would need a fresh weight per (token, position); the RNN reuses one cell — its inductive bias is "the rule for updating memory is the same at every timestep."

</div>

---

# Worked forward: "I love deep learning"

Tiny model · input dim $2$, hidden dim $3$, $h_0=[0,0,0]^\top$. Toy embeddings: $x_1=[1,0]^\top$ (I), $x_2=[0,1]^\top$ (love).

$$W_{xh}=\begin{pmatrix}0.1&0.2\\0.3&0.4\\0.5&0.6\end{pmatrix},\quad W_{hh}=\begin{pmatrix}0.1&0.2&0.3\\0.4&0.5&0.6\\0.7&0.8&0.9\end{pmatrix}$$

**Step 1 · "I".** $h_0=0$, so $h_1=\tanh(W_{xh}x_1)=\tanh([0.1,0.3,0.5]^\top)\approx[0.10,0.29,0.46]^\top$.

**Step 2 · "love".** Same $W_{xh},W_{hh}$, new input, previous state:
$$h_2=\tanh\big(\underbrace{[0.2,0.4,0.6]^\top}_{W_{xh}x_2}+\underbrace{[0.21,0.46,0.72]^\top}_{W_{hh}h_1}\big)=\tanh([0.41,0.86,1.32]^\top)\approx[0.39,0.70,0.87]^\top$$

Same weights — but $h_2$ now carries **both** words. That reuse is the whole RNN.

---

# The recurrence at a glance

Continue the identical pattern for all four tokens:

| step | input | update | $h_t$ encodes |
|:-:|:-:|:-:|:-:|
| 1 | $x_1$ | $\tanh(W_{xh}x_1 + W_{hh}h_0)$ | "I" |
| 2 | $x_2$ | $\tanh(W_{xh}x_2 + W_{hh}h_1)$ | "I love" |
| 3 | $x_3$ | $\tanh(W_{xh}x_3 + W_{hh}h_2)$ | "I love deep" |
| 4 | $x_4$ | $\tanh(W_{xh}x_4 + W_{hh}h_3)$ | full summary |

The readout $\hat y_t = W_{hy}h_t$ turns any hidden state into a prediction — and *where* you tap it off gives the four I/O shapes on the next slide.

---

# Four shapes, one cell

![w:920px](figures/lec10/svg/rnn_three_patterns.svg)

<div class="columns">
<div>

- **many-to-one** — sequence → one label (sentiment)
- **one-to-many** — vector → sequence (image captioning)

</div>
<div>

- **many-to-many, synced** — one label per token (POS tagging)
- **many-to-many, enc–dec** — sequence → sequence, different lengths (translation → **L13**)

</div>
</div>

Same recurrence throughout — you only change *where you read $\hat y_t$ off*.

---

# An RNN is a language model

Point the readout at a **softmax over the vocabulary** and the RNN scores the next token given all previous ones — the L01 NLL story with a *sequential* conditioning vector:

$$p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t}),\qquad p(x_t\mid x_{<t})=\text{softmax}(W_{hy}h_{t-1})$$

Train it by **teacher forcing**: feed the *true* previous token as $x_t$ and minimize per-step cross-entropy $-\sum_t \log p(x_t^\star\mid x_{<t}^\star)$ — the same loss as every classifier, summed over positions.

<div class="insight">

Nothing new in the loss — only the conditioning vector $h_{t-1}$ is now *recurrent*. This is exactly how a char-level RNN learns to spell, and the seed of every autoregressive LLM.

</div>

---

# Sampling a sequence

At inference there is no ground truth to feed — so **feed the model its own output**:

$$\hat x_t \sim p(\cdot\mid \hat x_{<t}) \;\longrightarrow\; \text{append}\;\longrightarrow\; \text{re-run the cell}$$

A **temperature** $\tau$ on the logits controls the gamble: $\tau\to 0$ is greedy/repetitive, large $\tau$ is creative/incoherent.

<div class="notebook">

**📓 Notebook · autoregressive generation** — train a small next-token RNN, then sample from it and watch coherence rise as it learns. *(ML · `notebooks/autoregressive-model.ipynb`)*

</div>

<div class="popquiz">

**Look ahead.** Training always saw the *true* previous token; sampling sees its *own* (possibly wrong) one. What could go wrong once an error slips in? *(the "exposure bias" we revisit in L13.)*

</div>

---

# The RNN cell in PyTorch — by hand

```python
class RNNCell(nn.Module):
    def __init__(self, d_in, d_h):
        super().__init__()
        self.W = nn.Linear(d_in, d_h, bias=False)   # W_xh
        self.U = nn.Linear(d_h,  d_h, bias=True)     # W_hh + b

    def forward(self, x_t, h_prev):
        return torch.tanh(self.W(x_t) + self.U(h_prev))

h = torch.zeros(batch, d_h)
for t in range(seq_len):          # the unrolled loop
    h = cell(x[:, t], h)
```

<div class="notebook">

**📓 Notebook · RNN by hand** — build this cell from `nn.Linear` only, run the unrolled loop, and check it matches `nn.RNNCell`. *(ES 667 · `notebooks/10-rnn-by-hand.ipynb`; language-model version: ML `notebooks/autoregressive-model.ipynb`)*

</div>

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A scalar RNN has $h_t=\tanh(w_{hh}\,h_{t-1}+w_{xh}\,x_t+b)$ with $w_{hh}=0.5$, $w_{xh}=1.0$, $b=0$, and $h_0=0$. The input stream is $x_1=1,\ x_2=1$. Compute $h_1$ and $h_2$ by hand.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**Step 1.** $h_1=\tanh(0.5\cdot 0 + 1.0\cdot 1)=\tanh(1)\approx \mathbf{0.762}$.

**Step 2.** $h_2=\tanh(0.5\cdot 0.762 + 1.0\cdot 1)=\tanh(1.381)\approx \mathbf{0.881}$.

<div class="keypoint">

The **same** $w_{hh},w_{xh},b$ produced both steps — and $h_2$ depends on $x_1$ *only through* $h_1$. That single channel is the memory; in Part 2 we watch the gradient try to travel back down it.

</div>

---

<!-- _class: section-divider -->

## Part 2 · BPTT & the vanishing gradient

---

# Unrolled = one very deep net

Training reuses **backpropagation** — but on the unrolled graph, which is $T$ copies of the *same* cell stacked in time. That is **Backpropagation Through Time (BPTT)**.

<div class="insight">

**Chinese whispers.** The gradient is a secret whispered backward from the loss to step 1. At every hop it is multiplied by the *same* Jacobian. After 20 hops it is either garbled to silence (**vanished**) or screaming (**exploded**) — because you multiplied by the same factor 20 times.

</div>

A 50-word sentence is a **50-layer** net where every layer shares one weight matrix. Depth in *time* has the same gradient pathology as depth in *layers* — only sharper, because the matrix repeats.

---

# The gradient is a product of Jacobians

Push the loss at step $T$ back to an early state $h_k$. The chain rule strings together one Jacobian per step:

$$\frac{\partial \mathcal L}{\partial h_k}=\frac{\partial \mathcal L}{\partial h_T}\prod_{t=k+1}^{T}\underbrace{\frac{\partial h_t}{\partial h_{t-1}}}_{W_{hh}^\top\,\text{diag}(\tanh')}$$

![w:600px](figures/lec10/svg/bptt_vanishing.svg)

The **same** $W_{hh}$ appears $T-k$ times — a matrix raised (roughly) to a power. Powers of a matrix either decay to $0$ or blow up, governed by its largest singular value.

---

# The product shrinks — or explodes

Strip it to a scalar: $h_t=\tanh(w\,h_{t-1})$, so $\frac{\partial h_t}{\partial h_{t-1}}=w\,\tanh'(\cdot)$ with $\tanh'\le 1$.

<div class="math-box">

$$\frac{\partial h_T}{\partial h_0}=\prod_{t=1}^{T} w\,\tanh'(\cdot)\ \le\ w^{T}$$

With $w=0.5$: &nbsp; $0.5^{10}\approx 10^{-3}$ &nbsp;·&nbsp; $0.5^{50}\approx 10^{-15}$ (**vanished**).
With $w=1.5$: &nbsp; $1.5^{50}\approx 6\times 10^{8}$ (**exploded**).

</div>

<div class="keypoint">

Two failure modes, one cause. $\|W_{hh}\|<1 \Rightarrow$ **vanishing** (can't learn long-range) · $\|W_{hh}\|>1 \Rightarrow$ **exploding** (loss → NaN). The knife-edge $\|W_{hh}\|=1$ is vanishingly unlikely to hold across training.

</div>

---

# Worked numeric · multiply by $W_{hh}$, $T$ times

The scalar picture is real for matrices too — the **largest eigenvalue** plays the role of $w$. Take a concrete diagonal $W_{hh}$ so the algebra is transparent:

$$W_{hh}=\begin{pmatrix}0.9&0\\0&1.2\end{pmatrix}\qquad\Rightarrow\qquad W_{hh}^{\,T}=\begin{pmatrix}0.9^{T}&0\\0&1.2^{T}\end{pmatrix}$$

<div class="math-box">

Push a gradient back $T=50$ steps (ignore the $\tanh'\le 1$ factor, which only *helps* it vanish):
$$0.9^{50}\approx 5\times 10^{-3}\ (\textbf{direction 1: vanishes}) \qquad 1.2^{50}\approx 9{,}100\ (\textbf{direction 2: explodes})$$

</div>

<div class="insight">

**One matrix, both diseases at once.** Along the eigenvector with $|\lambda|<1$ the signal dies; along the one with $|\lambda|>1$ it blows up. A general $W_{hh}$ also *rotates* (off-diagonal terms), but the rule is unchanged: raise the eigenvalues to the $T$. "Vanishing/exploding" is just *"multiply by $W_{hh}$, $T$ times."*

</div>

---

# Worked numeric · a 3-hop gradient by hand

Take the scalar RNN from Practice problem 1 ($w_{hh}=0.5$, stream $x_1{=}x_2{=}x_3{=}1$) and follow the gradient back **3 real steps**, keeping the $\tanh'$ factor this time. The states are $h_1\approx0.762,\ h_2\approx0.881,\ h_3\approx0.894$, and $\tanh'(z_t)=1-\tanh^2(z_t)=1-h_t^2$:

$$\frac{\partial h_t}{\partial h_{t-1}}=w_{hh}\,\tanh'(z_t)=0.5\,(1-h_t^2)$$

<div class="math-box">

$$\frac{\partial h_3}{\partial h_0}=\underbrace{0.5(1-0.894^2)}_{\approx\,0.100}\times \underbrace{0.5(1-0.881^2)}_{\approx\,0.112}\times \underbrace{0.5(1-0.762^2)}_{\approx\,0.210}\ \approx\ \mathbf{2.4\times10^{-3}}$$

</div>

<div class="insight">

Three hops already shrank the gradient **~400×** — and notice $\tanh'$ made it *worse* than the $0.5^3=0.125$ bound, because $\tanh'<1$ everywhere the state is nonzero. The saturating nonlinearity doesn't just fail to help; it actively *accelerates* vanishing. Now imagine 50 hops.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** A vanilla RNN's backward pass multiplies the gradient by a factor $\approx W_{hh}$ at every step. For a dependency spanning $T=30$ steps with effective scalar factor $0.7$, how much of the gradient survives? What if the factor were $1.2$? What does each imply for training?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

$$0.7^{30}\approx 2.3\times 10^{-5}\ (\textbf{vanished}) \qquad 1.2^{30}\approx 237\ (\textbf{exploded})$$

<div class="insight">

At $0.7$ the step-30 signal is $\sim\!10^{-5}$ of the recent-token signal — the net simply **can't learn** the long-range link; it optimizes only the last few tokens. At $1.2$ one update can overflow to NaN. This exponential-in-$T$ product is *why* vanilla RNNs rarely reach past a few tens of steps — and the entire motivation for the LSTM.

</div>

---

# Fix for explosion — gradient clipping

Exploding is the easier evil: rescale the gradient whenever its global norm exceeds a threshold, keeping the *direction*, shrinking the *step*.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

<div class="keypoint">

Pascanu et al. (2013): clipping at norm $\approx 1$ makes recurrent training robust. Cheap insurance against one bad batch erasing days of progress — still the default for **every** sequence model, RNN through Transformer.

</div>

Clipping tames explosion. It does **nothing** for vanishing — for that we need a new architecture.

---

# Fix for cost — truncated BPTT

Full BPTT stores **every** $h_t$ for the backward pass → activation memory grows $O(T)$. For thousand-step streams, backprop only $K$ steps at a time:

```python
for chunk in sequence.split(K, dim=1):
    h = h.detach()                 # cut the graph here → truncate
    for t in range(chunk.size(1)):
        h = cell(chunk[:, t], h)
    loss = criterion(h, target); loss.backward(); opt.step()
```

<div class="notebook">

**🎛 Interactive · vanishing gradients** — slide the sequence length and $\|W_{hh}\|$ and watch $\|\partial\mathcal L/\partial h_t\|$ decay across time. *(Interactive Lab · `interactive/articles/vanishing-gradients` · [live](https://nipunbatra.github.io/interactive-articles/vanishing-gradients/))*

</div>

Typical $K\!=\!32$–$256$: caps compute *and* memory at $O(K)$, at the cost of very-long-range gradient signal.

---

<!-- _class: section-divider -->

## Part 3 · LSTM — the additive gradient highway

---

# The idea: a protected memory belt

A vanilla RNN crams everything into one overwritten $h_t$. The **LSTM** (Hochreiter & Schmidhuber, 1997) adds a separate **cell state $c_t$** — a conveyor belt that carries information *unchanged* unless a gate intervenes.

![w:640px](figures/lec10/svg/lstm_annotated.svg)

<div class="insight">

Three learned controllers guard the belt: a **forget** gate (the janitor — erase), an **input** gate (the gatekeeper — write), an **output** gate (the press secretary — read). Long-term memory lives on the belt; the gates just schedule access.

</div>

---

# Two memories, not one

The single most common LSTM confusion: it carries **two** state vectors, with different jobs.

<div class="columns">
<div>

**Cell state $c_t$** — the *private* long-term memory. Lives on the additive belt, protected from the $\tanh$ crush. This is the channel gradients ride home.

</div>
<div>

**Hidden state $h_t$** — the *public* working output, $h_t=o_t\odot\tanh(c_t)$. What the next layer and the readout actually see, and what feeds the gates at $t+1$.

</div>
</div>

<div class="keypoint">

A vanilla RNN had to make **one** vector do both jobs — remember *and* report — so remembering long meant overwriting constantly. Splitting them is the LSTM's structural move; the gates are how the two states talk.

</div>

---

# What each gate outputs

Each gate is a **0–1 mask**, one value per memory slot, deciding *keep / write / read*. Same recipe for all three: squash $[h_{t-1},x_t]$ through a **sigmoid**, so every value lands in $[0,1]$.

<div class="columns">
<div>

- **forget $f_t$** — *what do I erase?* $0$ = flush · $1$ = keep
- **input $i_t$** — *how much new info?* $0$ = ignore · $1$ = accept

</div>
<div>

- **output $o_t$** — *what do I expose?* $0$ = silent · $1$ = project out
- **candidate $\tilde c_t$** — the new info on offer (a $\tanh$, in $[-1,1]$)

</div>
</div>

<div class="keypoint">

Get the **masks** first: sigmoid makes the dial, the dial multiplies the memory. The equations next are only "which vector each dial acts on."

</div>

---

# Intuition · the belt as editable memory

Picture the cell state $c_t$ as a **conveyor belt** running left to right through time. Boxes of information ride it forward. The three gates are **valves** the network learned to open and close:

<div class="columns">
<div>

- **forget** $f_t$ — the **erase** valve: how much of each box on the belt to wipe.
- **input** $i_t\odot\tilde c_t$ — the **write** valve: what new box to drop onto the belt.
- **output** $o_t$ — the **read** valve: which boxes to copy off the belt into $h_t$.

</div>
<div>

A concrete slot: belt carries $c_{t-1}=5$. This step $f_t=0.9$ (**keep 90%**), and write $i_t\odot\tilde c_t=+0.3$:
$$c_t = 0.9(5) + 0.3 = 4.8$$
The box rode forward almost untouched — the belt *defaults to remembering*.

</div>
</div>

<div class="keypoint">

The belt is **edited, not overwritten.** A vanilla RNN rebuilds $h_t$ from scratch each step (multiply-through); the LSTM only *nudges* the belt with valves. That "default to keep" is exactly why, next, the gradient can ride the belt home.

</div>

---

# The gate algebra: 3 sigmoids + 1 tanh

Stack $[h_{t-1},x_t]$ into one control vector; four little linear layers read it:

<div class="math-box">

$$f_t=\sigma(W_f[h_{t-1},x_t]+b_f)\qquad i_t=\sigma(W_i[h_{t-1},x_t]+b_i)$$
$$o_t=\sigma(W_o[h_{t-1},x_t]+b_o)\qquad \tilde c_t=\tanh(W_c[h_{t-1},x_t]+b_c)$$

</div>

![w:820px](figures/lec10/svg/lstm_cell.svg)

Three sigmoids → masks in $[0,1]$; one $\tanh$ → candidate in $[-1,1]$. That is the **entire** LSTM parameter set.

---

# The cell-state update — the "+" that saves everything

Keep some old memory, add some new — element-wise:

<div class="math-box">

$$c_t=\underbrace{f_t\odot c_{t-1}}_{\text{kept}}+\underbrace{i_t\odot\tilde c_t}_{\text{written}}\qquad\qquad h_t=o_t\odot\tanh(c_t)$$

</div>

<div class="keypoint">

The gates are just dials. The load-bearing symbol is the **+**: the cell state is *added to*, not matrix-multiplied through. Memory that isn't explicitly forgotten simply **persists** — and, next slide, so does its gradient.

</div>

---

# Why the "+" works: the gradient highway

Compare the backward path through one step:

<div class="columns">
<div>

**Vanilla RNN** — multiplicative:
$$\frac{\partial h_t}{\partial h_{t-1}}=W^\top\text{diag}(\tanh')$$
a **matrix** at every hop → product → vanish/explode.

</div>
<div>

**LSTM cell state** — additive:
$$\frac{\partial c_t}{\partial c_{t-1}}=f_t$$
just the **forget gate**, element-wise. No matrix, no $\tanh$ in the path.

</div>
</div>

![w:840px](figures/lec10/svg/additive_vs_multiplicative.svg)

If the net learns $f_t\approx 1$ ("don't forget"), the gradient flows back **unchanged**. Same trick as a ResNet skip — but across *time* instead of depth.

---

# Intuition · start the highway *open*

The highway only helps if $f_t$ is *near 1* early in training — but a freshly initialized net has $f_t\approx\sigma(0)=0.5$, halving the gradient every step ($0.5^{100}\approx 0$). The net can't learn to remember if it forgets before it ever learns.

<div class="keypoint">

**The one-line trick:** initialize the forget-gate **bias** $b_f=1$, so $f_t\approx\sigma(1)\approx 0.73$ *before any training*. The belt starts biased toward **keep**, the gradient flows on day one, and the net can *then* learn what to forget.

</div>

<div class="insight">

This is the LSTM analogue of "**start a ResNet as the identity**." You don't ask gradient descent to first *discover* that memory should persist — you make persistence the **default** and let it learn the exceptions. A tiny bias change; a huge difference on long-range tasks. *(Full account in the appendix.)*

</div>

---

# Gradient survival over 100 steps

Memory that must persist → the net learns $f_t\approx 0.99$. Contrast the vanilla factor $0.5$:

![w:480px](figures/lec10/svg/gradient_survival.svg)

$$0.99^{100}\approx \mathbf{0.37}\ (\text{usable}) \qquad\text{vs}\qquad 0.5^{100}\approx 8\times 10^{-31}\ (\text{below float precision})$$

<div class="notebook">

**🎛 Interactive · LSTM gates** — drag forget/input/output sliders and watch the cell state freeze, flow, or reset on a real sentence. *(Interactive Lab · `interactive/articles/lstm-gates` · [live](https://nipunbatra.github.io/interactive-articles/lstm-gates/))*

</div>

---

# Worked numeric — a single-neuron LSTM

1-D state. $c_{t-1}=5.0$ ("subject is **plural**"), input $x_t=$ "was". The net has learned, for this input:

- **forget** $f_t=0.1$ — sees "was" → drop the plural memory
- **input** $i_t=0.9$ — write the new fact aggressively
- **candidate** $\tilde c_t=-1.0$ — encodes "subject is **singular**"

$$c_t=f_t\,c_{t-1}+i_t\,\tilde c_t = 0.1(5.0)+0.9(-1.0)=0.5-0.9=\mathbf{-0.4}$$

The memory **flips** from strongly positive (plural) to negative (singular) in one step — precisely because forget was small *and* input was large. Grammar agreement, learned as gate behavior.

---

# Worked forward · one full LSTM step

That last example *handed* you the gates. Here we start one layer back — from the four **pre-activations** the linear layers emit — and run the sigmoids and $\tanh$ ourselves. 1-D cell, $c_{t-1}=0.8$:

<div class="columns">
<div>

**Gates** (squash each pre-activation):
$$f_t=\sigma(2.2)\approx 0.90$$
$$i_t=\sigma(-0.85)\approx 0.30$$
$$o_t=\sigma(0.85)\approx 0.70$$
$$\tilde c_t=\tanh(0.62)\approx 0.55$$

</div>
<div>

**Belt update**, then **read-out**:
$$c_t=f_t c_{t-1}+i_t\tilde c_t$$
$$=0.90(0.8)+0.30(0.55)\approx \mathbf{0.885}$$
$$h_t=o_t\tanh(c_t)$$
$$=0.70\,\tanh(0.885)\approx \mathbf{0.50}$$

</div>
</div>

<div class="insight">

Read the story off the numbers: a **high** $f$ kept most of the belt, a **low** $i$ let only a trickle of new info in, and the **output** valve exposed 70% of the (squashed) belt as $h_t$. Every LSTM step is just these six lines — three sigmoids, one $\tanh$, one weighted sum, one read.

</div>

---

# Reading the gates — what an LSTM learns

Because each gate is an interpretable $[0,1]$ dial per slot, you can *watch* a trained LSTM and find cells that specialize:

- a cell that stays **on inside quotes** and flips off at the closing `"` — a quote tracker
- a cell whose activation **ramps across a line** — a position/line-length counter
- a cell that toggles on **`(` … `)`** — bracket depth

<div class="insight">

Nobody programmed these — they *emerged* because the additive cell state lets a slot hold a value for hundreds of steps without decay. The gates learned *when* to open. (Karpathy et al., 2015, "Visualizing and Understanding RNNs.")

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** An LSTM's cell-state update is $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$. (a) Show $\partial c_t/\partial c_{t-1}=f_t$. (b) If a task needs memory over $T=100$ steps and the net learns $f_t=0.98$ on the relevant slot, estimate the surviving gradient. (c) Contrast with a vanilla RNN whose per-step factor is $0.9$.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

**(a)** In $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$, only the first term depends on $c_{t-1}$, and $f_t,i_t,\tilde c_t$ are treated as the gate outputs along this path → $\dfrac{\partial c_t}{\partial c_{t-1}}=f_t$ (element-wise). No $W$, no $\tanh'$.

**(b)** $\displaystyle\prod_{t}f_t \approx 0.98^{100}\approx \mathbf{0.13}$ — a *usable* gradient after 100 steps.

**(c)** Vanilla: $0.9^{100}\approx 2.7\times 10^{-5}$ — effectively **zero**.

<div class="keypoint">

Both factors are below $1$, yet only the LSTM survives — because $f_t$ is a **learned, near-1** scalar on a clean additive path, not a fixed matrix wrapped in $\tanh$. *Learning what to forget* is what keeps memory alive.

</div>

---

<!-- _class: section-divider -->

## Part 4 · GRU & when RNNs still win

---

# GRU — the lighter sibling

Cho et al. (2014) asked two questions: do we need a *private* $c_t$ **and** a *public* $h_t$? And since forget/input are often opposites, can **one** gate pick "old vs new"?

<div class="math-box">

$$z_t=\sigma(W_z[h_{t-1},x_t])\quad r_t=\sigma(W_r[h_{t-1},x_t])\quad \tilde h_t=\tanh(W[r_t\odot h_{t-1},x_t])$$
$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

</div>

One state, one **update gate** $z_t$ interpolating old↔new, one **reset gate** $r_t$ filtering the past into the candidate. The **additive** interpolation is, once again, the gradient highway — same fix, fewer parts.

---

# Intuition · the update gate is a blend knob

The GRU's whole update is a **crossfader** between the old memory and a fresh proposal:

$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

<div class="columns">
<div>

- $z_t\to 0$ → $h_t\approx h_{t-1}$: **copy the past.** This is the carry lane — gradient rides through untouched.
- $z_t\to 1$ → $h_t\approx\tilde h_t$: **overwrite** with the new candidate, like a fresh MLP layer.
- $z_t=0.5$ → a clean average of the two.

</div>
<div>

**One knob, not two.** The LSTM lets forget and input move independently (keep *and* write, or neither). The GRU **ties** them: $(1-z_t)$ and $z_t$ sum to 1, so *what you write, you must forget in equal measure.* One fewer gate, almost always the same accuracy.

</div>
</div>

<div class="keypoint">

Same fix as the LSTM belt, re-expressed: the **additive** blend $(1-z)\,h_{t-1}+z\,\tilde h$ is a gradient highway. $z_t$ small keeps the lane open; the reset $r_t$ only shapes the *proposal*, never the carry.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** A 1-D GRU has $h_{t-1}=2.0$. On this step the net emits reset $r_t=0.1$, update $z_t=0.2$, and (after forming the candidate) $\tilde h_t=-1.0$. (a) What does $r_t=0.1$ do to the past *inside* the candidate? (b) Compute $h_t=(1-z_t)h_{t-1}+z_t\tilde h_t$. (c) Redo (b) with $z_t=0.9$ and say what the gate chose.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

**(a)** The candidate sees $r_t\odot h_{t-1}=0.1\times 2.0=0.2$ — the reset nearly **wipes the past** when *proposing* $\tilde h_t$, so the new content is driven mostly by $x_t$. (The carry lane still keeps the old memory separately, via $1-z_t$.)

**(b)** $h_t=(1-0.2)(2.0)+0.2(-1.0)=1.6-0.2=\mathbf{1.4}$ — mostly the old memory, barely nudged.

**(c)** $h_t=(1-0.9)(2.0)+0.9(-1.0)=0.2-0.9=\mathbf{-0.7}$ — a near-total **overwrite**.

<div class="keypoint">

$z_t$ *is* the keep/overwrite dial: small $z_t$ carries memory forward (and gradient with it); large $z_t$ replaces it. Reset shapes the proposal; update decides how much of that proposal actually lands.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5.** Along the GRU carry lane the per-step Jacobian is dominated by $\partial h_t/\partial h_{t-1}\approx(1-z_t)$ (same move as the LSTM's $f_t$). A slot must hold memory for $T=100$ steps and the net learns $z_t=0.02$ on it. (a) Estimate the surviving gradient. (b) Contrast a vanilla RNN with per-step factor $0.9$. (c) Which LSTM quantity is $(1-z_t)$ playing the role of?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

**(a)** $\displaystyle\prod_{t}(1-z_t)\approx 0.98^{100}\approx \mathbf{0.13}$ — a usable gradient after 100 steps.

**(b)** Vanilla RNN: $0.9^{100}\approx 2.7\times10^{-5}$ — effectively **zero**.

**(c)** $(1-z_t)$ is the GRU's stand-in for the LSTM's **forget gate $f_t$**: a learned, near-1 scalar on a clean **additive** carry path.

<div class="insight">

Same cure, two spellings. LSTM: $\partial c_t/\partial c_{t-1}=f_t$. GRU: $\partial h_t/\partial h_{t-1}\approx 1-z_t$. Both replace the vanilla RNN's *fixed matrix wrapped in $\tanh$* with a **learned gate on an additive path** — which is *the* reason either one reaches past a few tens of steps.

</div>

---

# LSTM vs GRU — pick either

| | LSTM | GRU |
|---|---|---|
| gates | 3 + candidate | 2 + candidate |
| state | cell $c_t$ + hidden $h_t$ | hidden $h_t$ only |
| params ($d_h{=}128$) | $\approx 131$k | $\approx 98$k |
| speed | baseline | $\sim 15\%$ faster |
| accuracy | baseline | often tied |

<div class="insight">

The late-2010s ablation — *"we tried LSTM and GRU and kept whichever worked"* — is the real lesson: **the additive-gated recurrence matters; which gates you pick barely does.** Both crush the vanilla RNN on long-range tasks.

</div>

---

# Two knobs you'll actually turn

**Bidirectional** — run one RNN left→right and another right→left, concatenate. Uses *future* context, so it fits **tagging / classification**, never generation.

**Stacked (deep)** — feed layer $\ell$'s hidden states as inputs to layer $\ell+1$; higher layers capture coarser structure.

```python
self.rnn = nn.LSTM(d_in, d_h, num_layers=2,
                   bidirectional=True, batch_first=True)
```

Combined, **2-layer bidirectional LSTMs** were *the* NLP workhorse from 2015–2017 — until attention arrived.

---

# When RNNs still win

Transformers own language modeling, translation, and speech. But an RNN's $O(1)$-per-token, constant-memory update still wins where that matters:

<div class="columns">
<div>

**Streaming / online**
- process one token as it arrives
- no re-attending over all history
- constant latency per step

</div>
<div>

**Tiny devices**
- microcontrollers, always-on sensors
- KB-scale memory budgets
- an LSTM cell can be $\sim$1k params

</div>
</div>

<div class="insight">

And the recurrence is quietly back: **state-space models (Mamba)** and **linear RNNs (RWKV)** train in parallel yet run recurrently — Transformer quality at $O(1)$ inference. "Vanilla RNNs couldn't scale" ≠ "recurrence is dead."

</div>

---

# The bottleneck that tees up attention

Encoder-decoder translation compresses the **entire** source sentence into one fixed $h_t$:

> *"The animal didn't cross the street because **it** was too tired."*

To translate "it," the decoder must recover "animal" from a single vector that also had to store everything else. Longer input → more crammed in → more lost.

<div class="notebook">

**🎛 Interactive · the seq2seq bottleneck** — watch one fixed vector strain to hold a long sentence, and where it breaks. *(Interactive Lab · `interactive/articles/seq2seq-bottleneck`)*

</div>

**L13** builds this seq2seq encoder-decoder in full; **L14** removes the bottleneck with **attention** — the idea that let sequence models finally scale.

---

# See it in code — recurrence end to end

<div class="notebook">

**📓 Notebook · from cell to language model** — build the RNN cell from `nn.Linear`, run the unrolled loop, then train a next-token model and *sample* from it as coherence rises. *(ES 667 · `notebooks/10-rnn-by-hand.ipynb`; language-model version: ML · `notebooks/autoregressive-model.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · the three explainers, side by side** — **vanishing-gradients** (slide $T$ and $\|W_{hh}\|$, watch $\|\partial\mathcal L/\partial h_t\|$ decay), **lstm-gates** (drag forget/input/output, freeze or flush the belt), **seq2seq-bottleneck** (watch one vector strain to hold a long sentence). *(Interactive Lab · `interactive/articles/{vanishing-gradients, lstm-gates, seq2seq-bottleneck}`)*

</div>

<div class="notebook">

**🎛 Interactive · Mamba & the return of recurrence** — a **linear** state-space recurrence unrolls into a parallel scan: Transformer-style parallel training, RNN-style $O(1)$ inference — the additive-memory idea, made scalable. *(Interactive Lab · `interactive/articles/mamba`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**An RNN is an MLP that reuses one cell at every step and carries a memory forward — and the LSTM makes that memory *additive* so gradients survive.**

</div>

- **RNN** — weight sharing across time; $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b)$ is a running summary.
- **BPTT** — backprop on the unrolled net multiplies by $W_{hh}$ each step → **vanish / explode**, exponential in $T$.
- **Clipping** tames explosion; only a new architecture cures vanishing.
- **LSTM** — the additive $c_t=f\odot c_{t-1}+i\odot\tilde c_t$ gives $\partial c_t/\partial c_{t-1}=f$, a gradient highway; **GRU** does it with fewer gates.

**Next (L13):** two RNNs back-to-back — seq2seq — and the fixed-length bottleneck that makes attention inevitable.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **RNN cell, weight sharing, vanishing/exploding, LSTM gates** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`neural-networks/slides/rnn.tex`)
- **Autoregressive / language-model view** — ML `notebooks/autoregressive-model.ipynb`; ES 667 `notebooks/10-rnn-by-hand.ipynb`
- **Interactive explainers** — Interactive Lab · `lstm-gates`, `vanishing-gradients`, `seq2seq-bottleneck`, `mamba` (`nipunbatra.github.io/interactive-articles/`)
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization*, Course 5 (Sequence Models), Week 1.
- **The LSTM** — S. Hochreiter & J. Schmidhuber, *Long Short-Term Memory*, Neural Computation 9(8), 1997; GRU — Cho et al., 2014; gradient clipping — Pascanu et al., 2013.

Figures reused from the ES 667 figure library (`figures/lec10/`). All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · BPTT — the full Jacobian product

Loss $\mathcal L=\sum_t \mathcal L_t$. The gradient w.r.t. the shared $W_{hh}$ sums contributions from **every** step, and each routes back through all earlier states:

$$\frac{\partial \mathcal L}{\partial W_{hh}}=\sum_{t=1}^{T}\sum_{k=1}^{t}\frac{\partial \mathcal L_t}{\partial h_t}\left(\prod_{j=k+1}^{t}\frac{\partial h_j}{\partial h_{j-1}}\right)\frac{\partial h_k}{\partial W_{hh}}, \qquad \frac{\partial h_j}{\partial h_{j-1}}=W_{hh}^\top\,\text{diag}\big(\tanh'(z_j)\big)$$

Bound the inner product by submultiplicativity: $\big\|\prod_{j}\tfrac{\partial h_j}{\partial h_{j-1}}\big\|\le \big(\|W_{hh}\|\cdot\max\tanh'\big)^{t-k}$. Since $\max\tanh'=1$, the largest singular value $\sigma_{\max}(W_{hh})$ decides everything: $\sigma_{\max}<1 \Rightarrow$ terms with large $t-k$ **vanish**; $\sigma_{\max}>1 \Rightarrow$ they **explode**. The long-range terms (large $t-k$) are exactly the ones that die first — which is why plain BPTT learns short dependencies and forgets long ones.

*(Full treatment: Pascanu, Mikolov & Bengio, "On the difficulty of training RNNs," 2013.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the LSTM gradient path & modern recurrence

Along the cell-state channel the recurrence is $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$, so the exact Jacobian is $\text{diag}(f_t)$ (treating the gates' dependence on $c_{t-1}$ as second-order — the dominant term). Over a span,

$$\frac{\partial c_t}{\partial c_k}\approx\prod_{j=k+1}^{t}\text{diag}(f_j)$$

a product of **diagonal** matrices with entries in $[0,1]$ — no rotation, no $\tanh'$ shrinkage. Initializing $b_f>0$ (commonly $b_f=1$, so $\sigma(1)\approx 0.73$) biases gates *open* early, keeping the highway clear before the net has learned what to forget.

**Parallel-scan recurrence.** Drop the per-step nonlinearity entirely: $h_t=\bar A\,h_{t-1}+\bar B\,x_t,\ y_t=C h_t$. A **linear** recurrence unrolls into an associative scan → train in parallel like a Transformer, run recurrently at $O(1)$ inference. This is the core of **Mamba** (Gu & Dao, 2023) and **RWKV** (Peng, 2023) — the LSTM's additive-memory idea, made scalable.
