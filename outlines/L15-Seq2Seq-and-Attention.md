---
title: "Lecture 15: Sequence Models II"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 15: Sequence Models II  
## Encoder–Decoder, Teacher Forcing and Attention

Core story:

\[
\boxed{
\text{RNNs model one sequence. Encoder–decoder models map one sequence to another.}
}
\]

\[
\boxed{
\text{Attention removes the fixed-vector bottleneck.}
}
\]

---

## Part I — Why seq2seq?

### Slide 1 — Title

**Sequence Models II**

> Encoder–decoder, teacher forcing and attention

### Slide 2 — Tasks needing sequence output

\[
x_{1:T}\rightarrow y_{1:U}
\]

Examples:

- translation;
- summarization;
- speech recognition;
- captioning;
- dialogue response;
- time-series forecasting.

### Slide 3 — Input and output lengths differ

English:

```text
I am eating
```

Hindi:

```text
मैं खाना खा रहा हूँ
```

\[
T\
eq U
\]

Need flexible output length.

### Slide 4 — Conditional sequence probability

\[
p(y_{1:U}\mid x_{1:T})
=
\prod_{u=1}^{U}
p(y_u\mid y_{<u},x_{1:T})
\]

Same autoregressive idea, now conditioned on source sequence.

---

## Part II — Encoder–decoder without attention

### Slide 5 — Encoder

Read input:

\[
x_1,\ldots,x_T
\]

RNN encoder:

\[
h_t=f_{\text{enc}}(h_{t-1},e(x_t)).
\]

Final state:

\[
c=h_T.
\]

### Slide 6 — Decoder

Decoder generates output:

\[
s_u=f_{\text{dec}}(s_{u-1},e(y_{u-1}),c)
\]

\[
p(y_u\mid y_{<u},x)=\operatorname{softmax}(W_os_u+b_o).
\]

### Slide 7 — Architecture diagram

```text
x1 → x2 → x3 → x4
↓    ↓    ↓    ↓
ENC→ ENC→ ENC→ ENC
                 ↓ c
             DEC → DEC → DEC → DEC
              ↓     ↓     ↓     ↓
             y1    y2    y3    y4
```

### Slide 8 — Start and end tokens

Decoder input begins with:

\[
\texttt{<BOS>}
\]

Generation stops at:

\[
\texttt{<EOS>}
\]

Training pairs:

```text
decoder input:  <BOS> y1 y2 y3
target:          y1   y2 y3 <EOS>
```

### Slide 9 — Loss

\[
L
=
-\sum_{u=1}^{U}
\log p_\theta(y_u\mid y_{<u},x_{1:T})
\]

Masked for padding.

---

## Part III — Teacher forcing

### Slide 10 — During training

At decoder step \(u\), feed the true previous token:

\[
y_{u-1}^{\text{true}}
\]

not the model’s previous prediction.

This is called **teacher forcing**.

### Slide 11 — Why teacher forcing?

It gives stable supervised training:

\[
(\text{source},\ y_{<u}^{\text{true}})
\rightarrow y_u.
\]

Without it, early mistakes corrupt later inputs.

### Slide 12 — Training vs inference mismatch

Training:

\[
s_u=f(s_{u-1},y_{u-1}^{\text{true}},c)
\]

Inference:

\[
s_u=f(s_{u-1},\hat y_{u-1},c)
\]

This mismatch is called **exposure bias**.

### Slide 13 — Scheduled sampling

Idea:

Sometimes feed true previous token, sometimes feed model prediction.

\[
y_{u-1}^{\text{input}}
=
\begin{cases}
y_{u-1}^{\text{true}}, & \text{with prob }p\\
\hat y_{u-1}, & \text{with prob }1-p
\end{cases}
\]

Mention briefly; do not dwell.

---

## Part IV — Worked seq2seq example

### Slide 14 — Toy translation

Source:

```text
I like cats
```

Target:

```text
j'aime les chats
```

Input:

\[
x=[\texttt{I},\texttt{like},\texttt{cats},\texttt{<EOS>}]
\]

Decoder input:

\[
[\texttt{<BOS>},\texttt{j'aime},\texttt{les},\texttt{chats}]
\]

Target:

\[
[\texttt{j'aime},\texttt{les},\texttt{chats},\texttt{<EOS>}]
\]

### Slide 15 — Encoder states

\[
h_1,h_2,h_3,h_4
\]

Final context:

\[
c=h_4.
\]

### Slide 16 — Decoder step

At first step:

\[
s_1=f(s_0,e(\texttt{<BOS>}),c)
\]

\[
p_1=\operatorname{softmax}(W_os_1+b_o)
\]

Loss:

\[
L_1=-\log p_1(\texttt{j'aime})
\]

### Slide 17 — Full loss

\[
L=
-\log p_1(\texttt{j'aime})
-\log p_2(\texttt{les})
-\log p_3(\texttt{chats})
-\log p_4(\texttt{<EOS>})
\]

---

## Part V — Inference and decoding

### Slide 18 — Greedy decoding

\[
\hat y_u=\arg\max_v p(v\mid \hat y_{<u},x)
\]

Simple but locally greedy.

### Slide 19 — Beam search

Keep top \(B\) partial sequences.

At each step:

\[
\text{score}(y_{1:u})
=
\sum_{j=1}^{u}\log p(y_j\mid y_{<j},x)
\]

Expand and retain top \(B\).

### Slide 20 — Beam search diagram

```text
<BOS>
 ├─ a
 │  ├─ a b
 │  └─ a c
 └─ d
    ├─ d e
    └─ d f
```

Keep top beams by cumulative log score.

### Slide 21 — Length bias

Raw log-probability favours shorter sequences:

\[
\sum_u \log p(y_u\mid\cdot)
\]

because each term is \(\le0\).

Use length normalization:

\[
\frac{1}{U^\alpha}\sum_u\log p(y_u\mid\cdot)
\]

or similar.

---

## Part VI — Fixed-vector bottleneck

### Slide 22 — The bottleneck

Encoder compresses whole source into:

\[
c=h_T.
\]

For long sequences, one vector must store everything.

Problem:

\[
x_{1:T}\rightarrow c\rightarrow y_{1:U}
\]

### Slide 23 — What goes wrong?

Long input:

```text
The student who submitted the report after the deadline ...
```

The decoder may need early and late details.

A single \(c\) may be insufficient.

### Slide 24 — Desired behaviour

At each output step \(u\), decoder should look at relevant source positions.

For translating a word, attend to its corresponding source word.

This motivates attention.

---

## Part VII — Attention

### Slide 25 — Attention idea

Instead of one fixed vector \(c\), use all encoder states:

\[
h_1,\ldots,h_T.
\]

At decoder step \(u\), compute context:

\[
c_u=\sum_{t=1}^{T}\alpha_{ut}h_t.
\]

Weights:

\[
\alpha_{ut}\ge0,\qquad \sum_t\alpha_{ut}=1.
\]

### Slide 26 — Attention scores

Compare decoder state with each encoder state:

\[
e_{ut}=\operatorname{score}(s_{u-1},h_t).
\]

Then normalize:

\[
\alpha_{ut}
=
\frac{\exp(e_{ut})}
{\sum_{j=1}^{T}\exp(e_{uj})}.
\]

### Slide 27 — Common scoring functions

Dot product:

\[
e_{ut}=s_{u-1}^\top h_t
\]

General/bilinear:

\[
e_{ut}=s_{u-1}^\top W h_t
\]

Additive:

\[
e_{ut}=v^\top\tanh(W_ss_{u-1}+W_hh_t)
\]

Additive attention was introduced by Bahdanau et al.; Luong et al. later studied effective global/local attention variants for neural machine translation.

### Slide 28 — Context vector

\[
c_u=\sum_t\alpha_{ut}h_t.
\]

Then decoder predicts using:

\[
[s_u;c_u]
\]

\[
p(y_u\mid y_{<u},x)=
\operatorname{softmax}(W_o[s_u;c_u]+b_o).
\]

### Slide 29 — Attention diagram

```text
encoder states: h1 h2 h3 h4 h5
                  ↑  ↑  ↑  ↑  ↑
attention αu:    .1 .1 .6 .1 .1
                  ↓ weighted sum
                context cu
                  ↓
decoder step u predicts yu
```

### Slide 30 — Attention as differentiable lookup

Query:

\[
q=s_{u-1}
\]

Keys:

\[
k_t=h_t
\]

Values:

\[
v_t=h_t
\]

Scores:

\[
q^\top k_t
\]

Output:

\[
\sum_t\alpha_tv_t.
\]

This prepares for Transformers.

---

## Part VIII — Worked attention example

### Slide 31 — Encoder states

Suppose scalar encoder states:

\[
h=[1,2,4].
\]

Decoder query:

\[
q=2.
\]

Dot scores:

\[
e_t=qh_t.
\]

So:

\[
e=[2,4,8].
\]

### Slide 32 — Softmax weights

\[
\alpha_t=\frac{e^{e_t}}{\sum_j e^{e_j}}
\]

Numerically:

\[
\alpha\approx[0.0024,0.0179,0.9797].
\]

The model strongly attends to \(h_3\).

### Slide 33 — Context

\[
c=\sum_t\alpha_th_t
\]

\[
c\approx0.0024(1)+0.0179(2)+0.9797(4)
\]

\[
c\approx3.956.
\]

The context is close to \(h_3=4\).

### Slide 34 — Temperature intuition

If scores are scaled:

\[
\alpha=\operatorname{softmax}(e/\tau)
\]

Small \(\tau\): sharp attention.  
Large \(\tau\): diffuse attention.

Transformer scaled dot-product attention later uses:

\[
\frac{q^\top k}{\sqrt d}.
\]

---

## Part IX — Alignment and interpretability

### Slide 35 — Attention matrix

For decoder steps \(u=1,\ldots,U\) and source positions \(t=1,\ldots,T\):

\[
A\in\mathbb R^{U\times T}
\]

\[
A_{ut}=\alpha_{ut}.
\]

Visual: heatmap.

### Slide 36 — Alignment example

Source:

```text
I like cats
```

Target:

```text
j'aime les chats
```

Expected heatmap:

- `j'aime` attends to `like`;
- `chats` attends to `cats`.

Caution:

> Attention weights can be useful diagnostics, but they are not always faithful explanations.

### Slide 37 — Interactive 1: attention heatmap

`attention-heatmap.html`

Controls:

- source tokens;
- target step;
- editable scores;
- temperature.

Show:

- scores;
- softmax weights;
- context vector;
- heatmap.

---

## Part X — Seq2Seq with attention: full model

### Slide 38 — Encoder

\[
h_t=f_{\text{enc}}(h_{t-1},e(x_t)).
\]

Store all:

\[
H=[h_1,\ldots,h_T].
\]

### Slide 39 — Decoder with attention

At step \(u\):

\[
s_u=f_{\text{dec}}(s_{u-1},e(y_{u-1}),c_{u-1})
\]

Scores:

\[
e_{ut}=\operatorname{score}(s_u,h_t)
\]

Weights:

\[
\alpha_{ut}=\operatorname{softmax}_t(e_{ut})
\]

Context:

\[
c_u=\sum_t\alpha_{ut}h_t
\]

Output:

\[
p(y_u)=\operatorname{softmax}(W_o[s_u;c_u]+b_o).
\]

### Slide 40 — Full diagram

```text
source:  x1 → x2 → x3 → x4
         ↓    ↓    ↓    ↓
encoder: h1   h2   h3   h4
          ↘   ↓    ↓   ↙
          attention at each decoder step
                  ↓
decoder: <BOS> → y1 → y2 → y3
```

### Slide 41 — Training objective

\[
L
=
-\sum_{u=1}^{U}
\log p_\theta(y_u\mid y_{<u},x_{1:T})
\]

with attention computed at each decoder step.

Backprop flows through:

- decoder;
- attention scores;
- encoder states;
- encoder RNN.

---

## Part XI — Implementation view

### Slide 42 — Batching variable-length sequences

Need:

- padding;
- masks;
- packed sequences or attention masks.

Source mask:

\[
m_t=
\begin{cases}
1,&x_t\
eq\texttt{<PAD>}\\
0,&x_t=\texttt{<PAD>}
\end{cases}
\]

Attention should not attend to padding positions.

### Slide 43 — Masked attention

Set scores for padding to:

\[
-\infty
\]

before softmax:

\[
e_{ut}=
\begin{cases}
e_{ut},&m_t=1\\
-\infty,&m_t=0
\end{cases}
\]

Then:

\[
\alpha_{ut}=0
\]

for padding positions.

### Slide 44 — Teacher forcing code sketch

```python
decoder_input = BOS

for u in range(U):
    logits, state = decoder(decoder_input, state, encoder_outputs)
    loss += CE(logits, target[:, u])
    decoder_input = target[:, u]   # teacher forcing
```

### Slide 45 — Greedy decoding code sketch

```python
decoder_input = BOS

for u in range(max_len):
    logits, state = decoder(decoder_input, state, encoder_outputs)
    next_token = logits.argmax(dim=-1)
    outputs.append(next_token)
    decoder_input = next_token
```

### Slide 46 — Beam search code idea

Maintain:

\[
B
\]

partial hypotheses.

Each hypothesis stores:

- token sequence;
- decoder state;
- cumulative log score;
- finished/not finished flag.

---

## Part XII — Problems and limitations

### Slide 47 — RNN seq2seq limitations

- sequential encoder;
- sequential decoder;
- difficult long dependencies;
- attention helps bottleneck but not all recurrence costs;
- hard to parallelize over target generation.

### Slide 48 — Attention solves one bottleneck

Before attention:

\[
x_{1:T}\rightarrow c
\]

After attention:

\[
x_{1:T}\rightarrow h_{1:T}
\]

\[
c_u=\sum_t\alpha_{ut}h_t
\]

Decoder gets source-dependent context at every step.

### Slide 49 — But attention still uses RNN decoder

Even with attention:

\[
s_u=f(s_{u-1},y_{u-1},c_{u-1})
\]

Target generation remains sequential.

This motivates self-attention and Transformers.

---

## Part XIII — Interactives and demos

### Slide 50 — Interactive 2: seq2seq teacher forcing

`seq2seq-teacher-forcing.html`

Show:

- source tokens;
- decoder inputs;
- targets;
- predicted tokens;
- train vs inference mode;
- exposure bias.

### Slide 51 — Interactive 3: beam search

`beam-search.html`

Controls:

- beam width;
- vocabulary probabilities;
- length penalty.

Show tree expansion and selected hypotheses.

### Slide 52 — Interactive 4: attention seq2seq

`seq2seq-attention.html`

Show:

- encoder states;
- decoder state;
- scores;
- attention weights;
- context vector;
- output distribution.

---

## Part XIV — Summary

### Slide 53 — What changed from RNN LM?

RNN language model:

\[
p(y_t\mid y_{<t})
\]

Seq2Seq:

\[
p(y_t\mid y_{<t},x_{1:T})
\]

Attention:

\[
p(y_t\mid y_{<t},\sum_i\alpha_{ti}h_i)
\]

### Slide 54 — Key equations

\[
p(y_{1:U}\mid x_{1:T})
=
\prod_u p(y_u\mid y_{<u},x)
\]

\[
c_u=\sum_t\alpha_{ut}h_t
\]

\[
\alpha_{ut}=\operatorname{softmax}_t(\operatorname{score}(s_u,h_t))
\]

### Slide 55 — Final mental model

\[
\boxed{
\text{Encoder = represent source}
}
\]

\[
\boxed{
\text{Decoder = generate target autoregressively}
}
\]

\[
\boxed{
\text{Teacher forcing = stable training}
}
\]

\[
\boxed{
\text{Attention = dynamic access to source states}
}
\]

### Slide 56 — Next lecture

**Transformers I**

Key question:

> Can we remove recurrence entirely and use attention as the main computation?

Topics:

- queries, keys, values;
- self-attention;
- causal mask;
- positional encodings;
- Transformer block.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Motivation | 1–4 | 6 min |
| Encoder–decoder | 5–9 | 8 min |
| Teacher forcing | 10–13 | 7 min |
| Worked seq2seq | 14–17 | 7 min |
| Decoding | 18–21 | 8 min |
| Bottleneck | 22–24 | 5 min |
| Attention | 25–30 | 11 min |
| Worked attention | 31–34 | 7 min |
| Alignment | 35–37 | 5 min |
| Full attention model | 38–41 | 7 min |
| Implementation | 42–46 | 6 min |
| Limitations + summary | 47–56 | 3–5 min |

---

# Interactives

1. `attention-heatmap.html`
2. `seq2seq-teacher-forcing.html`
3. `beam-search.html`
4. `seq2seq-attention.html`
5. `masked-attention.html`

# Notebooks

1. `01_seq2seq_data_pipeline.ipynb`
2. `02_encoder_decoder_no_attention.ipynb`
3. `03_teacher_forcing_vs_inference.ipynb`
4. `04_beam_search_from_scratch.ipynb`
5. `05_attention_mechanism_numpy.ipynb`
6. `06_seq2seq_attention_pytorch.ipynb`
