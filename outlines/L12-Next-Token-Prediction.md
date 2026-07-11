---
title: "Lecture 12: Next-Token Prediction with Embeddings and MLPs"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 12: Next-Token Prediction with Embeddings and MLPs

## Core story

\[
\boxed{
\text{A language model estimates a probability distribution over the next token.}
}
\]

Start without recurrence. Let students first understand:

- sequence probability;
- tokenization;
- embeddings;
- context windows;
- causal next-token training;
- autoregressive generation;
- cross-entropy and perplexity.

Target: **80 minutes, approximately 55 slides**.

---

# Part I — What is sequence modelling?

### Slide 1 — Title

**Next-Token Prediction**

> Tokens, embeddings and neural language models

---

### Slide 2 — From images to sequences

An image is usually represented as a fixed tensor:

\[
X\in\mathbb R^{H\times W\times C}.
\]

A sequence has ordered elements:

\[
x_1,x_2,\ldots,x_T.
\]

Examples:

- characters;
- words or subwords;
- audio samples;
- sensor measurements;
- events;
- DNA bases.

---

### Slide 3 — Order matters

Compare:

```text
dog bites man
man bites dog
```

Same tokens, different sequence.

Therefore:

\[
f(x_1,x_2,x_3)
\
eq
f(x_3,x_2,x_1).
\]

The representation must preserve order.

---

### Slide 4 — Common sequence tasks

#### Sequence classification

\[
x_{1:T}\rightarrow y
\]

Example: sentiment classification.

#### Token classification

\[
x_{1:T}\rightarrow y_{1:T}
\]

Example: named-entity recognition.

#### Sequence-to-sequence

\[
x_{1:T}\rightarrow y_{1:U}
\]

Example: translation.

#### Next-token prediction

\[
x_{1:t}\rightarrow x_{t+1}.
\]

---

### Slide 5 — Our first sequence task

Given:

```text
deep learning is
```

predict:

```text
?
```

The model outputs:

\[
p_\theta(x_{t+1}=v\mid x_{1:t})
\]

for every vocabulary token:

\[
v\in\mathcal V.
\]

---

# Part II — Probability of a sequence

### Slide 6 — Sequence probability

For sequence:

\[
x_{1:T}=(x_1,\ldots,x_T),
\]

the chain rule gives:

\[
p(x_{1:T})
=
p(x_1)
p(x_2\mid x_1)
p(x_3\mid x_1,x_2)
\cdots
p(x_T\mid x_{1:T-1}).
\]

Compactly:

\[
\boxed{
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
}
\]

---

### Slide 7 — Why next-token prediction is enough

If we can model every conditional:

\[
p_\theta(x_t\mid x_{<t}),
\]

then we have a model of the complete sequence:

\[
p_\theta(x_{1:T})
=
\prod_t
p_\theta(x_t\mid x_{<t}).
\]

So language modelling becomes repeated classification.

---

### Slide 8 — Sequence log-likelihood

Take logs:

\[
\log p_\theta(x_{1:T})
=
\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
\]

Negative log-likelihood:

\[
L(\theta)
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t}).
\]

This is token-level cross-entropy.

---

### Slide 9 — Every position supplies a training example

Sentence:

```text
deep learning is fun
```

Create:

| Input context | Target |
|---|---|
| `<BOS>` | deep |
| `<BOS> deep` | learning |
| `deep learning` | is |
| `learning is` | fun |
| `is fun` | `<EOS>` |

One sequence produces many supervised examples.

---

### Slide 10 — Causal prediction

When predicting \(x_t\), the model may see:

\[
x_1,\ldots,x_{t-1}
\]

but not:

\[
x_t,x_{t+1},\ldots
\]

This is a **causal** or **autoregressive** objective.

Diagram:

```text
x1 → predict x2
x1 x2 → predict x3
x1 x2 x3 → predict x4
```

---

# Part III — Tokens and vocabulary

### Slide 11 — Text must become integers

Vocabulary:

\[
\mathcal V
=
\{\texttt{<BOS>},\texttt{<EOS>},\texttt{deep},
\texttt{learning},\texttt{is},\texttt{fun}\}.
\]

Mapping:

\[
\texttt{deep}\mapsto2,
\qquad
\texttt{learning}\mapsto3.
\]

The model receives token IDs, not strings.

---

### Slide 12 — Tokenization choices

#### Character-level

```text
learning → l e a r n i n g
```

#### Word-level

```text
deep | learning | is | useful
```

#### Subword-level

```text
un | predict | able
```

Trade-off:

| Tokenization | Vocabulary | Sequence length |
|---|---:|---:|
| character | small | long |
| word | very large | short |
| subword | moderate | moderate |

Do not go deeply into BPE yet.

---

### Slide 13 — Special tokens

Common tokens:

\[
\texttt{<BOS>}
\quad
\texttt{<EOS>}
\quad
\texttt{<PAD>}
\quad
\texttt{<UNK>}.
\]

Uses:

- begin generation;
- stop generation;
- batch unequal-length sequences;
- represent unknown items.

---

### Slide 14 — Why token ID is not a scalar feature

Suppose:

\[
\texttt{cat}\mapsto17,
\qquad
\texttt{dog}\mapsto18.
\]

This does **not** mean:

\[
\texttt{dog}-\texttt{cat}=1
\]

semantically.

Token IDs are categorical indices.

---

# Part IV — One-hot representations and embeddings

### Slide 15 — One-hot vector

Vocabulary size:

\[
V=10{,}000.
\]

A token \(i\) can be represented as:

\[
o_i\in\mathbb R^V,
\]

where:

\[
(o_i)_j=
\begin{cases}
1,&j=i,\\
0,&j\
eq i.
\end{cases}
\]

Problem:

- high-dimensional;
- sparse;
- no similarity structure.

---

### Slide 16 — Learn a dense embedding

Embedding matrix:

\[
E\in\mathbb R^{V\times d}.
\]

Token \(i\) gets vector:

\[
e_i=E_{i,:}\in\mathbb R^d.
\]

Usually:

\[
d\ll V.
\]

Example:

\[
V=50{,}000,\qquad d=256.
\]

---

### Slide 17 — Embedding lookup is a linear operation

For one-hot vector \(o_i\):

\[
e_i=E^\top o_i
\]

under a column-vector convention.

This simply selects one row of \(E\).

Diagram:

```text
token ID i
    ↓
embedding table E
    ↓
row Ei
```

---

### Slide 18 — Worked embedding lookup

Let:

\[
E=
\begin{bmatrix}
1&0\\
0&1\\
1&1\\
-1&1
\end{bmatrix}.
\]

Vocabulary:

\[
[\texttt{I},\texttt{like},\texttt{cats},\texttt{dogs}].
\]

Then:

\[
e_{\texttt{I}}=[1,0]
\]

\[
e_{\texttt{like}}=[0,1].
\]

Context:

```text
I like
```

becomes:

\[
([1,0],[0,1]).
\]

---

### Slide 19 — Embeddings are parameters

Initially:

\[
E_{ij}\sim\text{small random distribution}.
\]

During training:

\[
E\leftarrow E-\eta\
abla_E L.
\]

Only rows corresponding to observed tokens receive direct lookup gradients in that training example.

---

### Slide 20 — Embedding geometry

After training, tokens used in similar contexts may develop similar embeddings.

Similarity:

\[
\operatorname{cos}(e_i,e_j)
=
\frac{e_i^\top e_j}
{\|e_i\|\,\|e_j\|}.
\]

Do not promise that every semantic relation emerges perfectly.

---

### Slide 21 — Interactive 1: embedding lookup

`embedding-lookup.html`

Allow students to:

- edit a small embedding matrix;
- select token IDs;
- see the corresponding vectors;
- view vectors in 2D;
- compute cosine similarity;
- perform one gradient update.

---

# Part V — Fixed-context language model

### Slide 22 — Approximate the full history

The true model conditions on:

\[
x_{<t}.
\]

Start with only the previous \(k\) tokens:

\[
p_\theta(x_t\mid x_{t-k:t-1}).
\]

For \(k=3\):

```text
deep neural networks → predict are
```

This is a neural fixed-context language model.

---

### Slide 23 — Sliding-window dataset

Sequence:

```text
the cat sat on the mat
```

For context length \(k=3\):

| Context | Target |
|---|---|
| the cat sat | on |
| cat sat on | the |
| sat on the | mat |

---

### Slide 24 — MLP language-model architecture

For context:

\[
(x_{t-k},\ldots,x_{t-1}),
\]

look up embeddings:

\[
e_{t-j}=E[x_{t-j}].
\]

Concatenate:

\[
c_t=
[e_{t-k};\ldots;e_{t-1}]
\in\mathbb R^{kd}.
\]

Hidden layer:

\[
h_t=\phi(W_hc_t+b_h).
\]

Logits:

\[
z_t=W_oh_t+b_o.
\]

Probabilities:

\[
p_t=\operatorname{softmax}(z_t).
\]

---

### Slide 25 — Architecture diagram

```text
token IDs
   ↓
embedding lookup
   ↓
[e1 ; e2 ; ... ; ek]
   ↓
MLP hidden layer
   ↓
V logits
   ↓
softmax
   ↓
next-token distribution
```

This is essentially the classic neural probabilistic language-model construction.

---

### Slide 26 — Why concatenate?

Suppose:

```text
dog bites man
```

versus:

```text
man bites dog
```

Concatenation preserves position:

\[
[e_{\texttt{dog}};e_{\texttt{bites}};e_{\texttt{man}}]
\
eq
[e_{\texttt{man}};e_{\texttt{bites}};e_{\texttt{dog}}].
\]

Averaging embeddings would lose this ordering:

\[
\frac13(e_1+e_2+e_3).
\]

---

### Slide 27 — Tensor shapes

Let:

\[
B=\text{batch size},
\quad
k=\text{context length},
\quad
d=\text{embedding dimension},
\quad
H=\text{hidden width}.
\]

Token IDs:

\[
X:B\times k.
\]

Embeddings:

\[
E[X]:B\times k\times d.
\]

Flatten:

\[
C:B\times kd.
\]

Hidden:

\[
H_t:B\times H.
\]

Logits:

\[
Z:B\times V.
\]

---

### Slide 28 — Parameter count

Embedding table:

\[
Vd.
\]

Hidden layer:

\[
H(kd)+H.
\]

Output layer:

\[
VH+V.
\]

Total:

\[
\boxed{
Vd+Hkd+H+VH+V
}
\]

The embedding and vocabulary-output matrices are often the largest components.

---

### Slide 29 — Worked parameter count

Let:

\[
V=20{,}000,
\quad
d=128,
\quad
k=4,
\quad
H=512.
\]

Embedding:

\[
20{,}000\cdot128
=
2.56\text{ M}.
\]

Hidden:

\[
512(4\cdot128)+512
=
262{,}656.
\]

Output:

\[
20{,}000\cdot512+20{,}000
=
10.26\text{ M}.
\]

Total:

\[
\approx13.08\text{ M parameters}.
\]

---

# Part VI — Fully worked forward example

### Slide 30 — Tiny vocabulary

Vocabulary:

\[
[\texttt{I},\texttt{like},\texttt{cats},\texttt{dogs}].
\]

Context:

```text
I like
```

Target:

```text
cats
```

Embedding dimension:

\[
d=2.
\]

---

### Slide 31 — Embedding lookup

Use:

\[
e_{\texttt{I}}=[1,0]
\]

\[
e_{\texttt{like}}=[0,1].
\]

Concatenate:

\[
c=
\begin{bmatrix}
1\\0\\0\\1
\end{bmatrix}.
\]

---

### Slide 32 — Hidden layer

Let:

\[
W_h=
\begin{bmatrix}
1&0&0&1\\
0&1&1&0
\end{bmatrix},
\qquad
b_h=0.
\]

Then:

\[
W_hc=
\begin{bmatrix}
2\\0
\end{bmatrix}.
\]

With ReLU:

\[
h=
\begin{bmatrix}
2\\0
\end{bmatrix}.
\]

---

### Slide 33 — Output logits

Let output weights produce:

\[
z=
\begin{bmatrix}
-2\\
0\\
2\\
1
\end{bmatrix}
\]

for:

\[
[\texttt{I},\texttt{like},\texttt{cats},\texttt{dogs}].
\]

Softmax denominator:

\[
Z=e^{-2}+e^0+e^2+e^1.
\]

Numerically:

\[
Z\approx11.24.
\]

---

### Slide 34 — Probabilities and loss

Approximately:

\[
p=
\begin{bmatrix}
0.012\\
0.089\\
0.657\\
0.242
\end{bmatrix}.
\]

The target is `cats`, so:

\[
L=-\log p_{\texttt{cats}}
\]

\[
L\approx-\log(0.657)\approx0.42.
\]

The model assigns the highest probability to the correct token.

---

### Slide 35 — Output gradient

For softmax + cross-entropy:

\[
\
abla_zL=p-y.
\]

Here:

\[
y=
\begin{bmatrix}
0\\0\\1\\0
\end{bmatrix}.
\]

Thus:

\[
\
abla_zL
\approx
\begin{bmatrix}
0.012\\
0.089\\
-0.343\\
0.242
\end{bmatrix}.
\]

Training increases the correct logit and decreases competing logits.

---

### Slide 36 — Embedding gradients

Backpropagation continues:

\[
\
abla_hL=W_o^\top\
abla_zL
\]

\[
\
abla_cL=W_h^\top
\left(
\
abla_hL\odot\operatorname{ReLU}'(W_hc)
\right).
\]

Split \(\
abla_cL\) into two pieces:

\[
\
abla_{e_{\texttt{I}}}L,
\qquad
\
abla_{e_{\texttt{like}}}L.
\]

Those embedding rows are updated.

---

### Slide 37 — Interactive 2: MLP language model

`mlp-language-model.html`

Display:

- context token IDs;
- selected embedding rows;
- concatenated context;
- hidden activations;
- logits;
- softmax probabilities;
- true target;
- CE loss;
- \(p-y\);
- one SGD update.

Use the exact tiny worked example from Slides 30–36.

---

# Part VII — Training objective and evaluation

### Slide 38 — Dataset loss

For fixed context length \(k\):

\[
L(\theta)
=
-\frac1{T-k}
\sum_{t=k+1}^{T}
\log
p_\theta
(x_t\mid x_{t-k:t-1}).
\]

For several sequences, average over all non-padding target positions.

---

### Slide 39 — Minibatch training

For minibatch \(\mathcal B\):

\[
L_{\mathcal B}
=
-\frac1{B}
\sum_{i\in\mathcal B}
\log p_\theta(y_i\mid c_i).
\]

Then:

```text
contexts → embedding → MLP → logits → CE
                                      ↓
                                  backprop
                                      ↓
                              update E, Wh, Wo
```

---

### Slide 40 — Padding and loss masks

For unequal-length sequences, pad:

```text
I like cats <EOS> <PAD>
deep learning is fun <EOS>
```

Do not include padding targets in the loss.

Mask:

\[
m_{it}=
\begin{cases}
1,&\text{real token},\\
0,&\text{padding}.
\end{cases}
\]

Masked loss:

\[
L=
-\frac{
\sum_{i,t}m_{it}\log p(x_{it})
}{
\sum_{i,t}m_{it}
}.
\]

---

### Slide 41 — Per-token NLL

Average negative log-likelihood:

\[
\operatorname{NLL}
=
-\frac1N
\sum_{i=1}^{N}
\log p_\theta(x_i\mid\text{context}_i).
\]

Lower is better.

---

### Slide 42 — Perplexity

\[
\operatorname{PPL}
=
\exp(\operatorname{NLL}).
\]

Intuition:

> The effective number of plausible next-token choices.

Examples:

\[
\operatorname{NLL}=\log 2
\Rightarrow
\operatorname{PPL}=2.
\]

\[
\operatorname{NLL}=\log 10
\Rightarrow
\operatorname{PPL}=10.
\]

Perplexity should not be compared casually across different tokenizations.

---

# Part VIII — Autoregressive generation

### Slide 43 — Training versus generation

During training:

```text
true previous tokens → predict next token
```

During generation:

```text
generated previous tokens → predict next token
```

Generation proceeds one token at a time.

---

### Slide 44 — Autoregressive generation algorithm

1. Begin with context.
2. Compute next-token probabilities.
3. Select/sample a token.
4. Append it to context.
5. Drop oldest token if context is fixed.
6. Repeat until `<EOS>` or maximum length.

---

### Slide 45 — Greedy decoding

Choose:

\[
x_{t+1}
=
\arg\max_v p(v\mid x_{\le t}).
\]

Advantages:

- deterministic;
- simple.

Problem:

- may become repetitive;
- locally best token may not create globally best sequence.

---

### Slide 46 — Sampling

Sample:

\[
x_{t+1}\sim
\operatorname{Categorical}(p_t).
\]

This produces diverse generations.

Different runs may produce different sequences.

---

### Slide 47 — Temperature

Modify logits:

\[
p_i(\tau)
=
\frac{\exp(z_i/\tau)}
{\sum_j\exp(z_j/\tau)}.
\]

For:

\[
\tau<1
\]

distribution becomes sharper.

For:

\[
\tau>1
\]

distribution becomes flatter.

As:

\[
\tau\rightarrow0,
\]

sampling approaches greedy decoding.

---

### Slide 48 — Top-\(k\) sampling

Keep only the \(k\) highest-probability tokens.

Renormalize:

\[
\tilde p_i
=
\frac{p_i\mathbf 1[i\in\text{top-}k]}
{\sum_{j\in\text{top-}k}p_j}.
\]

This prevents extremely unlikely tokens from being sampled.

Keep nucleus sampling for the later Transformer/generation lecture.

---

### Slide 49 — Interactive 3: sampling laboratory

`next-token-sampling.html`

Controls:

- logits;
- temperature;
- greedy/sample;
- top-\(k\);
- random seed.

Show:

- original distribution;
- transformed distribution;
- sampled token;
- generated sequence over several steps.

---

# Part IX — What does the MLP model fail to do?

### Slide 50 — Fixed context

The model only sees:

\[
x_{t-k:t-1}.
\]

Information before \(t-k\) is invisible.

Example:

```text
The book that I placed on the tables near the window ... was
```

A short context may lose the grammatical subject.

---

### Slide 51 — Context length increases parameters

Input dimension:

\[
kd.
\]

Hidden matrix:

\[
W_h\in\mathbb R^{H\times kd}.
\]

Increasing \(k\) increases parameters:

\[
Hkd.
\]

The model cannot naturally accept arbitrary context length.

---

### Slide 52 — Position-specific parameters

After concatenation, position 1 and position 2 connect to different columns of \(W_h\).

The same token appearing at two context positions is processed differently:

\[
[e_a;e_b]
\
eq
[e_b;e_a].
\]

Order is preserved—but processing is not shared over time positions.

---

### Slide 53 — Desired sequence model

We want a model that:

- accepts variable-length sequences;
- shares parameters over time;
- carries information from earlier positions;
- produces an output at every step.

Idea:

\[
h_t=f(h_{t-1},x_t).
\]

This is a recurrent neural network.

---

### Slide 54 — From fixed window to recurrent state

Fixed-window MLP:

\[
p(x_{t+1}\mid x_{t-k:t}).
\]

RNN:

\[
h_t=f_\theta(h_{t-1},e_t)
\]

\[
p(x_{t+1}\mid x_{\le t})
=
\operatorname{softmax}(Wh_t+b).
\]

The hidden state becomes a learned summary of previous tokens.

---

### Slide 55 — Next lecture

**Sequence Models I**

Topics:

- recurrent neural networks;
- hidden state;
- unrolling through time;
- backpropagation through time;
- vanishing/exploding gradients;
- LSTMs;
- GRUs.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Sequence tasks | 1–5 | 6 min |
| Sequence probability | 6–10 | 9 min |
| Tokenization | 11–14 | 6 min |
| Embeddings | 15–21 | 11 min |
| MLP language model | 22–29 | 12 min |
| Fully worked example | 30–37 | 14 min |
| Training and evaluation | 38–42 | 8 min |
| Generation | 43–49 | 9 min |
| MLP limitations and RNN motivation | 50–55 | 5 min |

---

# Essential diagrams

1. Sequence task taxonomy.
2. Chain-rule factorization of sequence probability.
3. Sliding context-window construction.
4. One-hot vector versus embedding lookup.
5. Embedding table with selected rows highlighted.
6. Fixed-context MLP architecture.
7. Full worked forward/backward pipeline.
8. Training versus autoregressive generation.
9. Temperature transformation of token probabilities.
10. Fixed-context MLP versus recurrent-state model.

---

# Interactives

## 1. `sliding-window-builder.html`

Input editable text and controls:

- tokenizer: character/word;
- context length \(k\);
- include `<BOS>/<EOS>`.

Show generated context-target pairs.

## 2. `embedding-lookup.html`

Show:

- token IDs;
- one-hot vectors;
- embedding matrix;
- selected vectors;
- cosine similarities.

## 3. `mlp-language-model.html`

Main interactive:

\[
\text{context}
\rightarrow
\text{embeddings}
\rightarrow
\text{MLP}
\rightarrow
\text{softmax}
\rightarrow
L.
\]

## 4. `next-token-sampling.html`

Greedy, temperature, random sampling and top-\(k\).

## 5. `context-length.html`

Train/show predictions with:

\[
k=1,2,4,8.
\]

Demonstrate:

- information gained with larger context;
- parameter growth;
- inability to use earlier context.

---

# Notebooks

1. `01_tokenization_and_windows.ipynb`
2. `02_embedding_lookup_from_scratch.ipynb`
3. `03_mlp_language_model_numpy.ipynb`
4. `04_mlp_language_model_pytorch.ipynb`
5. `05_character_level_name_model.ipynb`
6. `06_generation_temperature_topk.ipynb`
7. `07_context_length_ablation.ipynb`

A small **character-level name generator** is ideal: tiny vocabulary, visible embeddings, fast training, and immediately satisfying generation.

---

# Subsequent lecture outlines

## Lecture 13 — Sequence Models I: RNNs, LSTMs and GRUs

Core flow:

1. Replace fixed context by hidden state:
   \[
   h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b)
   \]
2. Parameter sharing through time.
3. Unrolling an RNN.
4. Many-to-one, one-to-many and many-to-many tasks.
5. RNN language modelling:
   \[
   p(x_{t+1}\mid x_{\le t})=\operatorname{softmax}(W_oh_t)
   \]
6. Backpropagation through time.
7. Gradient products:
   \[
   \frac{\partial h_t}{\partial h_k}
   =
   \prod_{j=k+1}^{t}
   \frac{\partial h_j}{\partial h_{j-1}}
   \]
8. Vanishing and exploding gradients.
9. Gradient clipping.
10. LSTM cell state and gates.
11. GRU update/reset gates.
12. RNN versus LSTM versus GRU.
13. Worked numerical RNN step.
14. Interactive unrolled recurrence.
15. Sequence classification and next-token notebook.

LSTM was introduced specifically to address insufficient or decaying error flow over long time lags, while the GRU arose in the RNN encoder–decoder architecture as a simpler gated recurrent unit.

## Lecture 14 — Sequence Models II: Seq2Seq and Attention

Core flow:

1. Input and output can have different lengths.
2. Encoder RNN:
   \[
   h_t=f(h_{t-1},x_t)
   \]
3. Fixed-vector context:
   \[
   c=h_T.
   \]
4. Decoder:
   \[
   s_u=f(s_{u-1},y_{u-1},c).
   \]
5. Conditional factorization:
   \[
   p(y_{1:U}\mid x_{1:T})
   =
   \prod_u p(y_u\mid y_{<u},x_{1:T}).
   \]
6. Teacher forcing.
7. Shifted decoder inputs and targets.
8. Inference and autoregressive decoding.
9. Exposure bias.
10. Greedy versus beam search.
11. Fixed-vector bottleneck.
12. Attention scores:
    \[
    e_{ut}=\operatorname{score}(s_{u-1},h_t).
    \]
13. Attention weights:
    \[
    \alpha_{ut}=\operatorname{softmax}_t(e_{ut}).
    \]
14. Context:
    \[
    c_u=\sum_t\alpha_{ut}h_t.
    \]
15. Alignment heatmap.
16. Attention as differentiable retrieval.
17. Why attention motivates Transformers.

Sequence-to-sequence models used one recurrent network to encode an input sequence and another to decode its output; attention then replaced the single fixed context vector with a dynamically weighted combination of encoder states.

The dedicated MLP language-model lecture therefore improves the course flow substantially: students understand the **objective and data pipeline first**, then learn successive architectures for representing increasingly long contexts.
