---
title: "Lecture 14: Convolutional Sequence Models"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 14: Convolutional Sequence Models

## Causal CNNs, Dilated Convolutions, TCNs and WaveNet

Core story:

\[
\boxed{
\text{A sequence can be modelled without recurrence by convolving over time.}
}
\]

Target: **80 minutes, approximately 55 slides**

---

# Part I — Why another sequence architecture?

### Slide 1 — Title

**Convolutional Sequence Models**

> Causal convolutions, dilation, TCNs and WaveNet

---

### Slide 2 — Recall: the language-modelling objective

\[
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
\]

Training loss:

\[
L
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
\]

The objective does not require any particular architecture.

---

### Slide 3 — Architectures considered so far

#### Fixed-context MLP

\[
p(x_t\mid x_{t-k:t-1})
\]

#### RNN

\[
h_t=f(h_{t-1},x_t)
\]

\[
p(x_{t+1}\mid x_{\le t})=g(h_t).
\]

Today:

\[
p(x_{t+1}\mid x_{\le t})
=
g\!\left(\operatorname{CNN}(x_{\le t})\right).
\]

---

### Slide 4 — Why consider convolutions?

RNN limitation:

```text
h1 → h2 → h3 → ... → hT
```

Computation at \(t\) depends sequentially on \(t-1\).

A convolution can process several sequence positions simultaneously:

```text
x1 x2 x3 x4 x5 x6
 \____ convolution ____/
```

Convolutional sequence models permit parallel computation across time positions during training, unlike recurrent hidden-state updates.

---

### Slide 5 — Four questions

1. How does a 1D convolution operate on embeddings?
2. How do we prevent access to future tokens?
3. How can a CNN obtain a long context efficiently?
4. How does a convolutional model compare with an RNN?

---

# Part II — One-dimensional convolution over sequences

### Slide 6 — Sequence tensor

For batch size \(B\), length \(T\), embedding dimension \(D\):

\[
X\in\mathbb R^{B\times T\times D}.
\]

Interpretation:

- \(B\): examples;
- \(T\): time/token positions;
- \(D\): channels/features.

A temporal convolution slides along \(T\).

---

### Slide 7 — 1D convolution

For one input and one output channel:

\[
y_t
=
\sum_{i=0}^{k-1}w_i x_{t+i}.
\]

Here:

- \(k\): kernel size;
- the same \(w_i\) are reused at every time position.

---

### Slide 8 — Multi-channel temporal convolution

Input:

\[
X\in\mathbb R^{T\times C_{\text{in}}}.
\]

Kernel:

\[
W\in\mathbb R^{k\times C_{\text{in}}\times C_{\text{out}}}.
\]

Output:

\[
Y\in\mathbb R^{T'\times C_{\text{out}}}.
\]

Each output channel learns a temporal pattern over all input channels.

---

### Slide 9 — Worked scalar example

Sequence:

\[
x=[1,2,3,4,5].
\]

Kernel:

\[
w=[2,-1].
\]

Valid cross-correlation:

\[
y_1=2(1)-1(2)=0
\]

\[
y_2=2(2)-1(3)=1
\]

\[
y_3=2(3)-1(4)=2
\]

\[
y_4=2(4)-1(5)=3.
\]

Thus:

\[
y=[0,1,2,3].
\]

---

### Slide 10 — What can a temporal kernel detect?

A filter might respond to:

- rising or falling numerical trends;
- a local phoneme pattern;
- a word \(n\)-gram;
- a short motif in DNA;
- a local sensor event;
- nearby token combinations.

The filter is learned rather than manually designed.

---

### Slide 11 — Connection to the fixed-window MLP

A fixed-context MLP processes:

\[
[e_{t-k+1};\ldots;e_t]
\]

using a separate matrix over every relative position.

A temporal convolution processes every window using the **same kernel**.

\[
\boxed{\text{Convolution introduces temporal weight sharing.}}
\]

---

### Slide 12 — Translation equivariance in time

If a local pattern shifts in the sequence, its feature response also shifts:

\[
f(T_\Delta x)=T_\Delta f(x).
\]

The same language pattern can be detected at different positions.

---

### Slide 13 — Interactive 1: temporal convolution

`temporal-convolution.html`

Controls:

- editable scalar sequence;
- kernel size;
- kernel weights;
- padding;
- stride;
- multiple filters.

Show:

- sliding window;
- elementwise products;
- output feature sequence;
- filter response heatmap.

---

# Part III — Causal convolution

### Slide 14 — The future-information problem

For next-token prediction at position \(t\), the representation may depend only on:

\[
x_{\le t}.
\]

A centred convolution might use:

\[
x_{t+1},x_{t+2},
\]

which leaks future information.

---

### Slide 15 — Causal convolution

Define:

\[
\boxed{
y_t
=
\sum_{i=0}^{k-1}
w_i x_{t-i}
}
\]

Then \(y_t\) depends only on:

\[
x_t,x_{t-1},\ldots,x_{t-k+1}.
\]

No future input is used.

---

### Slide 16 — Causal connectivity diagram

For kernel size \(k=3\):

```text
x1  x2  x3  x4  x5
│ \ │ \ │
│  \│  \│
y1  y2  y3  y4  y5
```

At \(y_4\), connections are only to:

\[
x_2,x_3,x_4.
\]

---

### Slide 17 — Left padding

To preserve sequence length, pad only on the left:

\[
[\underbrace{0,\ldots,0}_{k-1},x_1,x_2,\ldots,x_T].
\]

For \(k=3\):

\[
[0,0,x_1,x_2,\ldots,x_T].
\]

Do not symmetrically pad for a causal model.

---

### Slide 18 — Shifted targets

Input:

```text
<BOS>  deep  learning  is
```

Targets:

```text
deep   learning  is    useful
```

Representation at input position \(t\) predicts target position \(t\).

The causal convolution prevents target leakage.

---

### Slide 19 — Strict causality versus inclusive causality

Two conventions:

#### Inclusive

\[
y_t=f(x_{\le t})
\]

used to predict:

\[
x_{t+1}.
\]

#### Strict

\[
y_t=f(x_{<t})
\]

used to predict:

\[
x_t.
\]

Both work if inputs and targets are shifted consistently.

---

### Slide 20 — Interactive 2: causal mask

`causal-convolution.html`

Show a time-by-time dependency matrix.

Controls:

- kernel size;
- number of layers;
- causal/noncausal;
- selected output position.

Highlight all input positions visible to the selected output.

---

# Part IV — Receptive field

### Slide 21 — One layer sees only a local window

A causal convolution with kernel \(k\) sees:

\[
R=k
\]

positions.

For:

\[
k=3,
\]

the model sees three tokens.

This remains a fixed-context model unless layers are stacked.

---

### Slide 22 — Stacking ordinary causal convolutions

With stride \(1\), kernel size \(k\), and \(L\) layers:

\[
\boxed{
R_L=1+L(k-1)
}
\]

For \(k=3\):

\[
R_L=1+2L.
\]

Ten layers see only:

\[
R_{10}=21
\]

positions.

---

### Slide 23 — Worked ordinary receptive field

For three \(k=3\) causal layers:

Layer 1:

\[
R_1=3.
\]

Layer 2:

\[
R_2=5.
\]

Layer 3:

\[
R_3=7.
\]

Diagram the dependency tree from one output backward to seven inputs.

---

### Slide 24 — Problem: linear receptive-field growth

To see \(1001\) positions with \(k=3\):

\[
1001=1+2L
\]

so:

\[
L=500.
\]

That is impractically deep.

We need a faster way to expand context.

---

# Part V — Dilated convolution

### Slide 25 — Dilation idea

Instead of adjacent inputs, skip positions.

Dilation \(d\):

\[
\boxed{
y_t
=
\sum_{i=0}^{k-1}
w_i x_{t-di}
}
\]

For \(k=3,d=2\):

\[
y_t
=
w_0x_t+w_1x_{t-2}+w_2x_{t-4}.
\]

---

### Slide 26 — Dilation diagrams

#### \(d=1\)

```text
x[t-2]  x[t-1]  x[t]
   \       |      /
           y[t]
```

#### \(d=2\)

```text
x[t-4]  x[t-2]  x[t]
   \       |      /
           y[t]
```

#### \(d=4\)

```text
x[t-8]  x[t-4]  x[t]
   \       |      /
           y[t]
```

---

### Slide 27 — Effective kernel width

For kernel size \(k\) and dilation \(d\):

\[
k_{\text{eff}}
=
1+(k-1)d.
\]

Example:

\[
k=3,\quad d=4
\]

gives:

\[
k_{\text{eff}}=1+2(4)=9.
\]

Only three parameters span nine positions.

---

### Slide 28 — Receptive field with varying dilation

For stride \(1\):

\[
\boxed{
R_L
=
1+
(k-1)
\sum_{\ell=1}^{L}d_\ell
}
\]

assuming one convolution per layer.

---

### Slide 29 — Exponentially increasing dilation

Choose:

\[
d_\ell=2^{\ell-1}.
\]

Then:

\[
\sum_{\ell=1}^{L}d_\ell
=
2^L-1.
\]

Therefore:

\[
\boxed{
R_L
=
1+(k-1)(2^L-1)
}
\]

For \(k=2\):

\[
R_L=2^L.
\]

---

### Slide 30 — Worked dilated receptive field

Use:

\[
k=3,
\qquad
d=[1,2,4,8].
\]

Then:

\[
R
=
1+2(1+2+4+8)
\]

\[
=
1+2(15)
=
31.
\]

Only four layers provide a 31-position context.

Ordinary convolutions would provide:

\[
1+4(2)=9.
\]

---

### Slide 31 — Dependency-tree visual

Draw four layers with dilations:

\[
1,2,4,8.
\]

Highlight how one top-layer output reaches a dense range of 31 past inputs.

Important visual message:

> Sparse connections at each layer combine to cover a dense historical interval.

---

### Slide 32 — Dilation cycle

For very deep models, repeat:

\[
1,2,4,8,1,2,4,8,\ldots
\]

This allows repeated local and long-range mixing rather than indefinitely increasing gaps.

WaveNet used stacks of causal dilated convolutions to achieve large receptive fields efficiently.

---

### Slide 33 — Interactive 3: dilation explorer

`dilated-receptive-field.html`

Controls:

- kernel size;
- layer count;
- dilation sequence;
- one/two convs per block;
- selected output position.

Display:

- dependency graph;
- receptive-field size;
- covered and uncovered input positions;
- comparison with ordinary convolution.

---

# Part VI — Temporal Convolutional Network

### Slide 34 — A practical TCN block

A typical TCN block contains:

```text
dilated causal conv
→ activation
→ dropout
→ dilated causal conv
→ activation
→ dropout
→ residual addition
```

TCNs combine causal convolutions, dilation, and residual blocks.

---

### Slide 35 — Residual TCN block

\[
z=F(x)
\]

\[
y=x+z.
\]

If channel dimensions differ:

\[
y=P(x)+F(x)
\]

using a \(1\times1\) projection.

The residual path helps train deep dilation stacks.

---

### Slide 36 — Preserve sequence length

Input:

\[
X\in\mathbb R^{T\times C_{\text{in}}}.
\]

Output:

\[
Y\in\mathbb R^{T\times C_{\text{out}}}.
\]

Causal padding ensures that every input position has a corresponding output position.

---

### Slide 37 — TCN for next-token prediction

Embeddings:

\[
E_t=E[x_t].
\]

TCN:

\[
H_{1:T}
=
\operatorname{TCN}(E_{1:T}).
\]

Output logits:

\[
z_t=W_oh_t+b_o.
\]

Prediction:

\[
p(x_{t+1}\mid x_{\le t})
=
\operatorname{softmax}(z_t).
\]

---

### Slide 38 — One forward pass predicts every position

During training:

\[
H_{1:T}
=
\operatorname{TCN}(E_{1:T})
\]

is computed in parallel.

Then:

\[
z_1,\ldots,z_T
\]

predict:

\[
x_2,\ldots,x_{T+1}.
\]

Causal structure ensures correctness even though positions are computed simultaneously.

---

### Slide 39 — TCN parameter sharing

Convolution weights are shared:

- across time positions;
- but not usually across network depth.

Contrast:

#### RNN

Same recurrent transition reused at every time step.

#### TCN

Same convolution reused at every position within a layer.

Different layers have different kernels.

---

### Slide 40 — Effective memory

The theoretical receptive field is:

\[
R.
\]

But effective dependence may be smaller if gradients or learned weights attenuate distant inputs.

TCN evaluations found that convolutional models could exhibit long effective memory on several sequence benchmarks, reinforcing that recurrence is not essential for sequence modelling.

---

# Part VII — WaveNet

### Slide 41 — WaveNet objective

For raw audio samples:

\[
x_1,\ldots,x_T,
\]

WaveNet models:

\[
p(x_{1:T})
=
\prod_t p(x_t\mid x_{<t}).
\]

It is a fully probabilistic autoregressive model operating directly on waveform samples.

---

### Slide 42 — Why audio is challenging

Typical audio sampling rates contain thousands of time steps per second.

A useful model needs:

- exact causality;
- large temporal context;
- fine local resolution;
- flexible output distribution.

Dilated causal convolutions suit this combination.

---

### Slide 43 — WaveNet gated activation

WaveNet used a gated activation of the form:

\[
z
=
\tanh(W_f*x)
\odot
\sigma(W_g*x).
\]

Interpretation:

- \(\tanh\) produces candidate content;
- sigmoid controls how much passes.

This resembles gating ideas from LSTMs but occurs inside convolutional blocks.

---

### Slide 44 — Residual and skip paths

A WaveNet block produces:

1. a residual output passed to the next block;
2. a skip output sent toward the final prediction head.

Diagram:

```text
input
  │
dilated causal conv
  │
tanh × sigmoid
  ├────────→ skip output
  │
1×1 conv
  │
  + ← residual input
  │
next block
```

---

### Slide 45 — WaveNet stack

Dilation pattern:

\[
1,2,4,8,\ldots,512
\]

then repeat.

Architecture:

```text
causal input conv
→ repeated dilated gated residual blocks
→ sum skip connections
→ ReLU
→ 1×1 conv
→ ReLU
→ 1×1 output logits
```

---

### Slide 46 — WaveNet training versus generation

#### Training

All positions can be processed in parallel because true previous samples are available.

#### Autoregressive generation

Generate one sample at a time:

\[
x_{t+1}\sim p(x_{t+1}\mid x_{\le t}).
\]

Generation remains sequential unless intermediate convolution states are cached.

---

### Slide 47 — WaveNet beyond audio

The architecture illustrates general principles useful for:

- time-series forecasting;
- sequence classification;
- language modelling;
- sensor modelling;
- biological sequences.

The central transferable ideas are causal convolution, dilation, residual blocks, and autoregressive prediction—not audio-specific output details.

---

### Slide 48 — Interactive 4: WaveNet block

`wavenet-block.html`

Controls:

- dilation;
- filter and gate kernels;
- input signal;
- residual scale;
- number of blocks.

Show:

- filter activation;
- gate activation;
- gated output;
- residual output;
- skip output;
- receptive field.

---

# Part VIII — CNN versus RNN

### Slide 49 — Context mechanism

#### RNN

\[
h_t=f(h_{t-1},x_t).
\]

Earlier information is compressed through recurrent state transitions.

#### TCN

\[
h_t=f(x_{t-R+1:t}).
\]

Earlier information is accessed through a finite convolutional receptive field.

---

### Slide 50 — Computational path length

Suppose information must travel from \(x_1\) to output \(y_T\).

#### RNN

Path length approximately:

\[
O(T).
\]

#### Dilated CNN

With exponentially growing dilation:

\[
O(\log T)
\]

layers may span the same interval.

This can improve gradient propagation between distant positions.

---

### Slide 51 — Parallelism

| Property | RNN | TCN |
|---|---:|---:|
| Parallel training across time | limited | high |
| Autoregressive generation | sequential | sequential |
| Context | recurrent state | receptive field |
| Context length | theoretically unbounded | explicitly finite |
| Temporal weight sharing | across steps | across positions |
| State/caching | hidden state | convolution cache |

---

### Slide 52 — Inductive bias

RNN bias:

\[
\text{sequential state update}
\]

CNN bias:

\[
\text{local temporal patterns + hierarchical composition}
\]

Neither is universally superior.

Choice depends on:

- sequence length;
- required context;
- latency;
- compute;
- dataset;
- task structure.

---

### Slide 53 — Why Transformers still become attractive

TCNs improve:

- parallel training;
- gradient path lengths;
- receptive-field growth.

But the context aggregation pattern is predetermined by the convolutional structure.

Attention instead allows each position to choose dynamically:

\[
\text{which previous positions matter?}
\]

That will become the key transition later.

---

# Part IX — Optional convolutional Seq2Seq preview

### Slide 54 — Can CNNs encode and decode sequences?

Yes.

Convolutional encoder:

\[
H=\operatorname{CNN}_{\text{enc}}(x_{1:T}).
\]

Causal convolutional decoder:

\[
s_u
=
\operatorname{CNN}_{\text{dec}}(y_{<u},H).
\]

ByteNet and convolutional Seq2Seq systems demonstrated that sequence-to-sequence architectures can be built predominantly or entirely from convolutions.

Keep this to one preview slide; teach encoder–decoder and attention properly in the next lecture.

---

### Slide 55 — Final summary

\[
\boxed{
\text{1D CNNs detect local temporal patterns.}
}
\]

\[
\boxed{
\text{Causal convolutions prevent future leakage.}
}
\]

\[
\boxed{
\text{Dilation grows context exponentially with depth.}
}
\]

\[
\boxed{
\text{TCNs combine causal dilation and residual blocks.}
}
\]

\[
\boxed{
\text{WaveNet applies these ideas to autoregressive generation.}
}
\]

Next:

> Mapping one sequence into another: encoder–decoder, teacher forcing and attention.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Motivation | 1–5 | 6 min |
| Temporal convolution | 6–13 | 11 min |
| Causality | 14–20 | 10 min |
| Receptive field | 21–24 | 6 min |
| Dilated convolution | 25–33 | 15 min |
| TCN | 34–40 | 10 min |
| WaveNet | 41–48 | 13 min |
| CNN versus RNN | 49–53 | 7 min |
| Seq2Seq preview and summary | 54–55 | 2 min |

---

# Essential worked examples

## 1. Ordinary temporal convolution

\[
x=[1,2,3,4,5],
\qquad
w=[2,-1].
\]

Compute every output explicitly.

## 2. Causal padding

For:

\[
k=3,
\]

show why the left padding must be:

\[
[0,0,x_1,\ldots,x_T].
\]

Demonstrate the information leakage caused by symmetric convolution.

## 3. Dilated receptive field

For:

\[
k=3,\qquad d=[1,2,4,8],
\]

derive:

\[
R
=
1+2(1+2+4+8)
=
31.
\]

Compare with ordinary convolution:

\[
R=9.
\]

## 4. TCN output shapes

Input:

\[
B\times T\times D.
\]

Embedding projection:

\[
D\rightarrow C.
\]

Four residual TCN blocks:

\[
B\times T\times C.
\]

Vocabulary head:

\[
B\times T\times V.
\]

Loss against shifted targets:

\[
B\times T.
\]

---

# Interactives

1. **`temporal-convolution.html`**  
   Sliding 1D kernels over scalar or embedding sequences.

2. **`causal-convolution.html`**  
   Causal versus noncausal dependency matrix and leakage demonstration.

3. **`dilated-receptive-field.html`**  
   Main interactive: configure dilation and inspect context coverage.

4. **`tcn-block.html`**  
   Two causal convolutions, activation, dropout and residual path.

5. **`wavenet-block.html`**  
   Gated activation, residual and skip outputs.

6. **`rnn-vs-tcn.html`**  
   Compare computation graphs, path length and parallel execution.

---

# Notebooks

1. `01_temporal_conv_from_scratch.ipynb`
2. `02_causal_padding_and_leakage.ipynb`
3. `03_dilated_receptive_fields.ipynb`
4. `04_tcn_language_model.ipynb`
5. `05_wavenet_block_pytorch.ipynb`
6. `06_rnn_vs_tcn_sequence_task.ipynb`
7. `07_character_level_tcn_generation.ipynb`

Use the **same character-level name dataset** as the MLP and RNN lectures. Then students can directly compare:

\[
\text{MLP LM}
\quad\text{vs}\quad
\text{RNN LM}
\quad\text{vs}\quad
\text{TCN LM}.
\]

The lecture is worthwhile, but I would not add a separate full lecture on convolutional Seq2Seq. Mention ByteNet and ConvS2S briefly here, then proceed to the more conceptually important **encoder–decoder and attention** lecture.
