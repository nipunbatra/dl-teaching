---
title: "Lecture 13: Sequence Models I"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 13: Sequence Models I  
## RNNs, BPTT, LSTMs and GRUs

Core story:

\[
\boxed{
\text{Fixed-window models remember by input context. RNNs remember by hidden state.}
}
\]

Use this after next-token prediction + embeddings. RNNs give dynamic memory; LSTMs and GRUs add gates to control remembering/forgetting. Elman-style recurrent networks were introduced as simple networks with dynamic memory, LSTM was designed to address vanishing/error-flow issues over long lags, and GRU was introduced in the RNN encoder–decoder line.

---

## Part I — Motivation

### Slide 1 — Title

**Sequence Models I**

> RNNs, backpropagation through time, LSTMs and GRUs

---

### Slide 2 — Recall: next-token prediction

\[
p(x_{1:T})=\prod_{t=1}^T p(x_t\mid x_{<t})
\]

Training loss:

\[
L=-\sum_t \log p_\theta(x_t\mid x_{<t})
\]

The objective is clear. Now we need a better architecture for context.

---

### Slide 3 — Fixed-window MLP limitation

MLP language model:

\[
p(x_t\mid x_{t-k:t-1})
\]

Problems:

\[
\boxed{\text{fixed context length}}
\]

\[
\boxed{\text{parameters grow with }k}
\]

\[
\boxed{\text{no shared state over time}}
\]

---

### Slide 4 — Desired property

We want a model that reads a sequence one token at a time:

\[
x_1,x_2,\ldots,x_T
\]

and maintains memory:

\[
h_t=\text{summary of }x_{1:t}.
\]

Then predict:

\[
p(x_{t+1}\mid x_{\le t})=g(h_t).
\]

---

### Slide 5 — The recurrent idea

\[
\boxed{
h_t=f_\theta(h_{t-1},x_t)
}
\]

Same function reused at every time step.

This gives:

- variable-length input;
- parameter sharing;
- a learned state;
- online processing.

---

## Part II — Vanilla RNN

### Slide 6 — RNN equations

Let input embedding be:

\[
e_t=E[x_t].
\]

Vanilla RNN:

\[
a_t=W_{xh}e_t+W_{hh}h_{t-1}+b_h
\]

\[
h_t=\phi(a_t)
\]

Output logits:

\[
z_t=W_{hy}h_t+b_y
\]

\[
p_t=\operatorname{softmax}(z_t)
\]

---

### Slide 7 — RNN cell diagram

```text
h_{t-1} ─┐
         ├→ affine → tanh/ReLU → h_t → output
x_t    ──┘
```

Same cell is applied for every \(t\).

---

### Slide 8 — Unrolled RNN

```text
x1      x2      x3      x4
↓       ↓       ↓       ↓
[RNN] → [RNN] → [RNN] → [RNN]
↓       ↓       ↓       ↓
h1      h2      h3      h4
```

Important:

\[
W_{xh},W_{hh},W_{hy}
\]

are shared across all time steps.

---

### Slide 9 — Hidden state as memory

Recursive expansion:

\[
h_t=f(h_{t-1},x_t)
\]

\[
h_t=f(f(h_{t-2},x_{t-1}),x_t)
\]

\[
h_t=f(f(\cdots f(h_0,x_1)\cdots),x_t)
\]

So \(h_t\) depends on all previous inputs.

---

### Slide 10 — RNN for language modelling

Input:

```text
<BOS> deep learning is
```

Targets:

```text
deep learning is useful
```

At each step:

\[
h_t=\operatorname{RNN}(h_{t-1},e_t)
\]

\[
p(x_{t+1}\mid x_{\le t})=\operatorname{softmax}(W_{hy}h_t+b_y)
\]

Loss:

\[
L=-\sum_t \log p_t(x_{t+1})
\]

---

### Slide 11 — RNN task patterns

| Pattern | Example | Output |
|---|---|---|
| many-to-one | sentiment | final \(h_T\) |
| many-to-many aligned | POS tagging | every \(h_t\) |
| language model | next token | every \(h_t\) |
| sequence-to-sequence | translation | encoder-decoder |

---

### Slide 12 — Shape slide

For batch size \(B\), length \(T\):

\[
X:B\times T
\]

Embeddings:

\[
E[X]:B\times T\times d
\]

Hidden states:

\[
H:B\times T\times m
\]

Logits:

\[
Z:B\times T\times V
\]

Targets:

\[
Y:B\times T
\]

Loss:

\[
\text{CE}(Z,Y)
\]

with padding mask if needed.

---

## Part III — Fully worked RNN forward pass

### Slide 13 — Tiny scalar RNN

Use scalar values:

\[
h_t=\tanh(w_xx_t+w_hh_{t-1}+b)
\]

Let:

\[
w_x=1,\quad w_h=0.5,\quad b=0,\quad h_0=0.
\]

Sequence:

\[
x_1=1,\quad x_2=2,\quad x_3=-1.
\]

---

### Slide 14 — Step 1

\[
a_1=1\cdot1+0.5\cdot0=1
\]

\[
h_1=\tanh(1)\approx0.762.
\]

---

### Slide 15 — Step 2

\[
a_2=1\cdot2+0.5\cdot0.762=2.381
\]

\[
h_2=\tanh(2.381)\approx0.983.
\]

---

### Slide 16 — Step 3

\[
a_3=1\cdot(-1)+0.5\cdot0.983=-0.509
\]

\[
h_3=\tanh(-0.509)\approx-0.469.
\]

Show table:

| \(t\) | \(x_t\) | \(a_t\) | \(h_t\) |
|---:|---:|---:|---:|
| 1 | 1 | 1.000 | 0.762 |
| 2 | 2 | 2.381 | 0.983 |
| 3 | -1 | -0.509 | -0.469 |

---

### Slide 17 — Why this is memory

At \(t=3\):

\[
h_3
=
\tanh(x_3+0.5h_2)
\]

and \(h_2\) contains information from \(x_1,x_2\).

So \(h_3\) is not just a function of \(x_3\).

---

### Slide 18 — Interactive 1: RNN forward unroll

`rnn-forward-unroll.html`

Controls:

- sequence values;
- \(w_x,w_h,b\);
- activation;
- \(h_0\).

Show:

- unrolled graph;
- \(a_t,h_t\);
- state trajectory over time;
- effect of changing \(w_h\).

---

## Part IV — Backpropagation through time

### Slide 19 — RNN is a deep network in time

Unrolled RNN:

```text
h0 → h1 → h2 → h3 → ... → hT
```

Training uses backpropagation through this unrolled graph.

This is called:

\[
\boxed{\text{Backpropagation Through Time (BPTT)}}
\]

---

### Slide 20 — Loss over time

For sequence prediction:

\[
L=\sum_{t=1}^{T} L_t
\]

where:

\[
L_t=-\log p_t(y_t).
\]

A parameter such as \(W_{hh}\) affects every future hidden state:

\[
W_{hh}\rightarrow h_t,h_{t+1},\ldots,h_T.
\]

---

### Slide 21 — Shared parameter gradients accumulate

Because \(W_{hh}\) is reused at every time step:

\[
\
abla_{W_{hh}}L
=
\sum_{t=1}^{T}
\
abla_{W_{hh}}L_t.
\]

Same idea as gradient accumulation in computation graphs.

\[
\boxed{\text{same parameter, many time uses, gradients add}}
\]

---

### Slide 22 — Backward recurrence

For vanilla RNN:

\[
a_t=W_{xh}e_t+W_{hh}h_{t-1}+b
\]

\[
h_t=\phi(a_t).
\]

Let:

\[
\bar h_t=\frac{\partial L}{\partial h_t}.
\]

Then:

\[
\bar a_t=\bar h_t\odot\phi'(a_t)
\]

\[
\
abla_{W_{xh}}L{+}{=}\bar a_t e_t^\top
\]

\[
\
abla_{W_{hh}}L{+}{=}\bar a_t h_{t-1}^\top
\]

\[
\bar h_{t-1}{+}{=}W_{hh}^\top\bar a_t.
\]

---

### Slide 23 — Two sources of \(\bar h_t\)

At time \(t\), hidden state contributes to:

1. current output loss \(L_t\);
2. future hidden state \(h_{t+1}\).

Thus:

\[
\bar h_t
=
\underbrace{
\frac{\partial L_t}{\partial h_t}
}_{\text{output path}}
+
\underbrace{
\frac{\partial L_{\ge t+1}}{\partial h_t}
}_{\text{future path}}.
\]

This is gradient accumulation.

---

### Slide 24 — Worked BPTT skeleton

For \(T=3\):

\[
L=L_1+L_2+L_3.
\]

Backward order:

```text
t=3: compute gradients from L3, pass to h2
t=2: combine gradient from L2 + future, pass to h1
t=1: combine gradient from L1 + future, pass to h0
```

Use arrows with accumulated gradients.

---

### Slide 25 — Truncated BPTT

Full BPTT over long sequences is expensive.

Truncated BPTT backpropagates only \(K\) steps:

\[
x_{1:T}
\rightarrow
\text{chunks of length }K.
\]

State may be carried forward, but gradient is stopped between chunks:

\[
h_{\text{start}}=\operatorname{detach}(h_{\text{start}}).
\]

---

### Slide 26 — Interactive 2: BPTT gradient flow

`rnn-bptt.html`

Show:

- unrolled RNN;
- losses at each time;
- gradients flowing backward;
- accumulation at shared \(W_{hh}\);
- truncation length \(K\).

---

## Part V — Vanishing and exploding gradients

### Slide 27 — Long-term gradient path

For:

\[
h_t=\phi(W_{hh}h_{t-1}+W_{xh}x_t),
\]

the gradient from \(h_T\) to \(h_k\) contains:

\[
\frac{\partial h_T}{\partial h_k}
=
\prod_{j=k+1}^{T}
\frac{\partial h_j}{\partial h_{j-1}}.
\]

Each factor is roughly:

\[
D_jW_{hh}
\]

where:

\[
D_j=\operatorname{diag}(\phi'(a_j)).
\]

---

### Slide 28 — Product problem

If typical multiplier magnitude is \(<1\):

\[
\left\|\frac{\partial h_T}{\partial h_k}\right\|\rightarrow0.
\]

If \(>1\):

\[
\left\|\frac{\partial h_T}{\partial h_k}\right\|\rightarrow\infty.
\]

This is the RNN version of vanishing/exploding gradients. Pascanu, Mikolov and Bengio analyze these difficulties and discuss gradient clipping for exploding gradients.

---

### Slide 29 — Scalar recurrence example

Let:

\[
h_t=\tanh(wh_{t-1})
\]

Near zero:

\[
\tanh'(0)=1.
\]

So:

\[
\frac{\partial h_T}{\partial h_0}
\approx
w^T.
\]

If:

\[
w=0.9,\quad T=50,
\]

\[
w^T\approx0.005.
\]

If:

\[
w=1.1,\quad T=50,
\]

\[
w^T\approx117.
\]

---

### Slide 30 — Gradient clipping

If gradient norm is too large:

\[
\|g\|_2 > c,
\]

replace:

\[
g\leftarrow c\frac{g}{\|g\|_2}.
\]

This preserves direction but caps magnitude.

Use for exploding gradients.

---

### Slide 31 — Worked clipping example

Let:

\[
g=[6,8].
\]

\[
\|g\|_2=10.
\]

Clip threshold:

\[
c=5.
\]

Then:

\[
g_{\text{clip}}
=
5\frac{[6,8]}{10}
=
[3,4].
\]

---

### Slide 32 — Clipping does not solve vanishing

Clipping handles:

\[
\|g\|\rightarrow\infty.
\]

It does not fix:

\[
\|g\|\rightarrow0.
\]

For long-term memory, we need architecture changes.

This motivates gated RNNs.

---

### Slide 33 — Interactive 3: vanishing/exploding gradient

`rnn-gradient-dynamics.html`

Controls:

- recurrent weight scale;
- sequence length;
- activation: tanh/ReLU/linear;
- clipping threshold.

Show:

- hidden state trajectory;
- gradient norm versus time lag;
- clipped versus unclipped gradient.

---

## Part VI — LSTM intuition

### Slide 34 — The core problem

Vanilla RNN updates:

\[
h_t=\phi(W_xx_t+W_hh_{t-1})
\]

Memory is overwritten every time step.

We want the model to decide:

- what to forget;
- what to write;
- what to expose.

---

### Slide 35 — LSTM has two states

Hidden/output state:

\[
h_t
\]

Cell/memory state:

\[
c_t
\]

The cell state has an additive update path:

\[
c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t.
\]

This additive path is key for better gradient flow.

---

### Slide 36 — Gates

Gates are vectors with entries in \([0,1]\).

Computed using sigmoid:

\[
g=\sigma(\cdot).
\]

Interpretation:

\[
0\Rightarrow\text{block}
\]

\[
1\Rightarrow\text{pass}
\]

LSTM uses gates to control memory. Hochreiter and Schmidhuber introduced LSTM to address insufficient/decaying error backflow over long time lags.

---

### Slide 37 — LSTM equations

Concatenate:

\[
u_t=[h_{t-1};x_t].
\]

Forget gate:

\[
f_t=\sigma(W_fu_t+b_f)
\]

Input gate:

\[
i_t=\sigma(W_iu_t+b_i)
\]

Candidate memory:

\[
\tilde c_t=\tanh(W_cu_t+b_c)
\]

Output gate:

\[
o_t=\sigma(W_ou_t+b_o)
\]

Cell update:

\[
c_t=f_t\odot c_{t-1}+i_t\odot \tilde c_t
\]

Hidden state:

\[
h_t=o_t\odot\tanh(c_t)
\]

---

### Slide 38 — LSTM cell diagram

```text
c_{t-1} ──× f_t ──┐
                  + ── c_t ── tanh ──× o_t ── h_t
candidate ─× i_t ─┘

[h_{t-1}, x_t] → gates f_t, i_t, o_t, candidate
```

Use colour:

- forget path;
- write path;
- output path.

---

### Slide 39 — Forget gate

\[
f_t=\sigma(W_fu_t+b_f)
\]

\[
f_t\odot c_{t-1}
\]

If:

\[
f_t=1
\]

memory is retained.

If:

\[
f_t=0
\]

memory is erased.

---

### Slide 40 — Input gate and candidate

Candidate:

\[
\tilde c_t=\tanh(W_cu_t+b_c)
\]

Input gate:

\[
i_t=\sigma(W_iu_t+b_i)
\]

Write amount:

\[
i_t\odot \tilde c_t.
\]

The model decides what new information to store.

---

### Slide 41 — Output gate

\[
h_t=o_t\odot\tanh(c_t)
\]

Cell state \(c_t\) may store information.

Hidden state \(h_t\) exposes part of it to:

- output head;
- next recurrent step.

---

### Slide 42 — Why additive memory helps

Cell transition:

\[
c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t.
\]

Derivative:

\[
\frac{\partial c_t}{\partial c_{t-1}}=f_t.
\]

If:

\[
f_t\approx1,
\]

then gradients can flow through \(c_t\) without repeated multiplication by arbitrary recurrent matrices.

This is the central LSTM advantage.

---

### Slide 43 — Worked scalar LSTM memory update

Let:

\[
c_{t-1}=5.
\]

Forget gate:

\[
f_t=0.8.
\]

Input gate:

\[
i_t=0.2.
\]

Candidate:

\[
\tilde c_t=3.
\]

Then:

\[
c_t=0.8(5)+0.2(3)=4+0.6=4.6.
\]

Memory mostly retained.

---

### Slide 44 — Another scalar example

If:

\[
f_t=0.1,\quad i_t=0.9,\quad \tilde c_t=-2,
\]

then:

\[
c_t=0.1(5)+0.9(-2)=0.5-1.8=-1.3.
\]

Memory mostly replaced.

---

### Slide 45 — Interactive 4: LSTM cell

`lstm-cell.html`

Controls:

- \(c_{t-1}\);
- \(f_t\);
- \(i_t\);
- \(\tilde c_t\);
- \(o_t\).

Show:

- memory retained;
- new memory written;
- \(c_t\);
- \(h_t\);
- \(\partial c_t/\partial c_{t-1}=f_t\).

---

## Part VII — GRU

### Slide 46 — GRU motivation

GRU is a simpler gated recurrent unit.

It merges ideas of:

- memory;
- hidden state;
- gating.

GRU has fewer gates than LSTM. It was introduced in the RNN encoder–decoder model for phrase representation and translation scoring.

---

### Slide 47 — GRU equations

Update gate:

\[
z_t=\sigma(W_zx_t+U_zh_{t-1}+b_z)
\]

Reset gate:

\[
r_t=\sigma(W_rx_t+U_rh_{t-1}+b_r)
\]

Candidate:

\[
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h)
\]

Final state:

\[
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
\]

---

### Slide 48 — GRU diagram

```text
h_{t-1} ──× (1-z_t) ──┐
                      + ── h_t
candidate ─× z_t ─────┘

reset gate r_t controls candidate computation
```

---

### Slide 49 — Update gate interpretation

\[
h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t.
\]

If:

\[
z_t=0
\]

then:

\[
h_t=h_{t-1}.
\]

If:

\[
z_t=1
\]

then:

\[
h_t=\tilde h_t.
\]

So \(z_t\) controls how much to update.

---

### Slide 50 — Reset gate interpretation

Candidate:

\[
\tilde h_t=\tanh(W_hx_t+U_h(r_t\odot h_{t-1})+b_h).
\]

If:

\[
r_t=0
\]

previous state is ignored while computing candidate.

If:

\[
r_t=1
\]

previous state is fully used.

---

### Slide 51 — Worked scalar GRU update

Let:

\[
h_{t-1}=4,\quad z_t=0.25,\quad \tilde h_t=0.
\]

Then:

\[
h_t=(1-0.25)4+0.25(0)=3.
\]

Mostly keeps old state.

If:

\[
z_t=0.9,
\]

then:

\[
h_t=0.1(4)+0.9(0)=0.4.
\]

Mostly replaces old state.

---

### Slide 52 — LSTM vs GRU

| Feature | LSTM | GRU |
|---|---|---|
| states | \(h_t,c_t\) | \(h_t\) |
| gates | forget, input, output | update, reset |
| memory path | explicit cell state | hidden state interpolation |
| parameters | more | fewer |
| common use | strong default | simpler/faster default |

Practical note:

\[
\boxed{\text{Try both; performance is task-dependent.}}
\]

---

### Slide 53 — Interactive 5: GRU cell

`gru-cell.html`

Controls:

- \(h_{t-1}\);
- update gate \(z_t\);
- reset gate \(r_t\);
- candidate \(\tilde h_t\).

Show:

- retained old state;
- candidate contribution;
- final \(h_t\);
- comparison with LSTM memory update.

---

## Part VIII — RNN language model and classification

### Slide 54 — Character-level RNN language model

Input:

```text
a l i c
```

Target:

```text
l i c e
```

At each step:

\[
p_t=\operatorname{softmax}(W_oh_t+b_o)
\]

Loss:

\[
L=-\sum_t\log p_t(x_{t+1})
\]

Generate by sampling:

\[
x_{t+1}\sim p_t.
\]

---

### Slide 55 — Sequence classification

For many-to-one classification:

\[
h_T=\operatorname{RNN}(x_{1:T})
\]

\[
p(y\mid x_{1:T})=\operatorname{softmax}(Wh_T+b).
\]

Alternative:

\[
h_{\text{pool}}=\frac1T\sum_t h_t
\]

then classify.

---

### Slide 56 — Bidirectional RNN

For non-causal tasks:

Forward:

\[
\overrightarrow h_t=f(\overrightarrow h_{t-1},x_t)
\]

Backward:

\[
\overleftarrow h_t=f(\overleftarrow h_{t+1},x_t)
\]

Combined:

\[
h_t=[\overrightarrow h_t;\overleftarrow h_t].
\]

Use when future context is allowed.

Not for autoregressive next-token generation.

---

### Slide 57 — Packed sequences and masks

Real sequences have different lengths.

Use padding:

```text
I like cats <EOS> <PAD>
deep learning is fun <EOS>
```

Loss mask:

\[
m_t=
\begin{cases}
1,&\text{real token}\\
0,&\text{padding}
\end{cases}
\]

Do not train on padding.

---

## Part IX — Practical training

### Slide 58 — RNN training loop

```python
h = init_hidden(batch_size)

for x, y in loader:
    h = h.detach()

    logits, h = model(x, h)
    loss = masked_ce(logits, y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    optimizer.step()
```

Key points:

- detach hidden state for truncated BPTT;
- mask padding;
- clip gradients.

---

### Slide 59 — Common bugs

| Symptom | Possible cause |
|---|---|
| loss NaN | exploding gradients, too high LR |
| model predicts PAD | padding not masked |
| memory blowup | not detaching hidden state |
| no long-term learning | vanilla RNN vanishing |
| generation repeats | sampling/temperature issue |
| train slow | sequence too long, no batching/packing |

---

### Slide 60 — RNN vs TCN preview

RNN:

\[
h_t=f(h_{t-1},x_t)
\]

TCN:

\[
h_t=f(x_{t-R+1:t})
\]

RNN carries state sequentially.  
TCN uses receptive field.

Next lecture compares convolutional sequence models.

---

## Part X — Summary

### Slide 61 — One-slide summary

\[
\boxed{\text{RNN}:\ h_t=f(h_{t-1},x_t)}
\]

\[
\boxed{\text{BPTT}:\ \text{backprop through unrolled time graph}}
\]

\[
\boxed{\text{Vanilla RNNs}:\ \text{vanishing/exploding gradients}}
\]

\[
\boxed{\text{LSTM}:\ \text{cell state + gates}}
\]

\[
\boxed{\text{GRU}:\ \text{simpler gated recurrent update}}
\]

---

### Slide 62 — Final mental model

\[
\boxed{
\text{RNNs remember by carrying a hidden state.}
}
\]

\[
\boxed{
\text{LSTMs/GRUs learn when to remember, forget and update.}
}
\]

\[
\boxed{
\text{Gates create controllable paths for information and gradients.}
}
\]

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Motivation | 1–5 | 6 min |
| Vanilla RNN | 6–12 | 10 min |
| Worked forward pass | 13–18 | 8 min |
| BPTT | 19–26 | 13 min |
| Gradient issues | 27–33 | 10 min |
| LSTM | 34–45 | 15 min |
| GRU | 46–53 | 10 min |
| Applications/training | 54–60 | 6 min |
| Summary | 61–62 | 2 min |

---

# Essential diagrams

1. Fixed-window MLP versus recurrent state.
2. RNN cell.
3. Unrolled RNN with shared parameters.
4. BPTT backward arrows.
5. Gradient accumulation at \(W_{hh}\).
6. Vanishing/exploding gradient curve versus time lag.
7. LSTM cell with gates.
8. GRU cell with update/reset gates.
9. Many-to-one and many-to-many usage patterns.
10. Padding/masking diagram.

---

# Interactives

1. `rnn-forward-unroll.html`  
   Scalar/vector RNN forward pass.

2. `rnn-bptt.html`  
   Unrolled graph, losses, shared-parameter gradient accumulation.

3. `rnn-gradient-dynamics.html`  
   Vanishing/exploding gradients and clipping.

4. `lstm-cell.html`  
   Forget/input/output gates and cell-state update.

5. `gru-cell.html`  
   Update/reset gates and interpolation.

6. `rnn-language-model.html`  
   Character-level name model generation.

---

# Notebooks

1. `01_scalar_rnn_forward.ipynb`
2. `02_rnn_from_scratch_numpy.ipynb`
3. `03_bptt_manual_small_rnn.ipynb`
4. `04_vanishing_exploding_rnn_gradients.ipynb`
5. `05_lstm_gru_cells_from_scratch.ipynb`
6. `06_character_rnn_language_model.ipynb`
7. `07_sequence_classification_lstm_gru.ipynb`
