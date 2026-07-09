---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Sequence Models II · Seq2Seq &amp; Word Embeddings

## Lecture 13 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

Two moves let neural nets *read one sequence and write another*. First, stop feeding words as meaningless indices — **learn a vector for each token so meaning lives in geometry.** Then wire two RNNs together: one **encodes** the source into a context vector, one **decodes** it into the target.

<div class="keypoint">

**A learned embedding puts meaning in geometry; an encoder–decoder puts a whole sentence into one vector — and *that one vector* is the wall that makes attention inevitable.**
`king − man + woman ≈ queen` is not magic — it is what happens when you train a lookup table by gradient descent. And `one vector for the whole sentence` is fine at 5 words and fatal at 50.

</div>

$$\underbrace{\text{one-hot} \to \text{embedding table}}_{\text{meaning as geometry}} \;\rightarrow\; \underbrace{\text{encoder} \to \mathbf c \to \text{decoder}}_{\text{seq2seq}} \;\rightarrow\; \underbrace{\text{the fixed-context bottleneck}}_{\text{why L14 needs attention}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L14** — attention *deletes* today's bottleneck; QKV is the fix
- **L15–L18** — the Transformer keeps the embedding table verbatim; only "context" changes
- **teacher forcing / next-token NLL** — the exact objective every LLM is still trained on
- **temperature sampling** — the decoding knob you'll meet again at inference

</div>
<div>

**Borrowed from your own courses** (we build on, not repeat):
- next-token classification &amp; the embedding table — **ML (ES 335)** (Karpathy *makemore*)
- cross-entropy / NLL as *the* loss — **L1** of this course
- RNN / LSTM hidden states — **L11–L12**

</div>
</div>

<div class="insight">

One thread ties it together: **the softmax-over-next-token loss from L1, wrapped around a sequence.** Embeddings feed it; seq2seq stacks two of them; attention (L14) fixes where it breaks.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Words need geometry — the embedding table

---

# One-hot is wasteful — and blind to meaning

To feed a word to a network we first turn it into a number. The naive choice is **one-hot**: a length-$V$ vector, a single 1.

<div class="columns">
<div>

$$\text{cat} = [0,0,1,0,\dots,0]$$
$$\text{dog} = [0,1,0,0,\dots,0]$$

</div>
<div>

- $V \approx 50{,}000$ → a 50k-dim vector per token
- every pair is **orthogonal**: $\langle\text{cat},\text{dog}\rangle = 0$
- "cat" is exactly as far from "dog" as from "airplane"

</div>
</div>

<div class="insight">

One-hot encodes *identity* but throws away *similarity*. The model has to relearn that "cat" and "dog" behave alike, separately, for every context. We want a representation where **similar words sit close together.**

</div>

---

# The fix: a learned embedding puts meaning in geometry

Replace the one-hot with a short **dense** vector — its coordinates are *learned*, so the network can place related words near each other.

<div class="columns">
<div>

**Embedding table** $E \in \mathbb R^{V \times d}$

| token | $d_1$ | $d_2$ | $\dots$ | $d_K$ |
|:-:|:-:|:-:|:-:|:-:|
| a | 0.2 | −0.1 | … | 0.8 |
| b | −0.3 | 0.5 | … | −0.2 |
| ⋮ | ⋮ | ⋮ | ⋱ | ⋮ |
| z | 0.7 | −0.4 | … | 0.1 |

</div>
<div>

Row $i$ *is* the embedding of token $i$. "Looking up" a token is just picking its row:

$$\text{one-hot}(i)^\top E = E_i$$

A one-hot times a matrix = **selecting a row.** So the embedding layer is a matrix multiply that a lookup shortcuts — and $E$ is **learnable**, updated by backprop like any weight.

</div>
</div>

---

# Intuition · meaning becomes geometry

Andrew Ng's one-liner for embeddings: **turn every word into a point in space, and let *distance* mean *similarity*.** Once meaning is geometry, the network gets neighbours, clusters, and analogies for free.

<div class="columns">
<div>

**One-hot is blind.** Every word is its own axis, all mutually perpendicular. "cat" is exactly as far from "dog" as from "airplane" — the representation *knows nothing* about meaning, so the model must relearn every similarity from scratch, in every context.

</div>
<div>

**Learned vectors share structure.** Coordinates are trained, so related words pack together and **directions carry relationships** — a "plural" direction, a "past-tense" direction, a "capital-of" direction. Structure one-hot could never express becomes a straight-line move.

</div>
</div>

<div class="insight">

The whole shift of Part 1: stop storing words as *identities* (one-hot) and start storing them as *positions* (dense vectors). Similar words sit close; a relationship is a direction you can add. Everything that follows — vowels clustering, `king−man+woman`, cosine scores — is a *consequence* of that one move.

</div>

---

# Worked example · a lookup *is* a row-select

Make the "one-hot × matrix = pick a row" identity concrete. Take a tiny table with $V=4$ tokens and $d=3$:

<div class="columns">
<div>

$$E = \begin{bmatrix} \mathbf{0.2} & \mathbf{-0.1} & \mathbf{0.8}\\ -0.3 & 0.5 & -0.2\\ 0.7 & -0.4 & 0.1\\ 0.0 & 0.9 & 0.6 \end{bmatrix}$$

Token **"cat"** = index 0 → one-hot $[1,0,0,0]$.

</div>
<div>

$$[1,0,0,0]\,E = [\,\mathbf{0.2},\,\mathbf{-0.1},\,\mathbf{0.8}\,] = E_0$$

The 1 selects row 0; the zeros erase every other row. No arithmetic actually happens — which is why frameworks skip the multiply and **index** the row directly (`nn.Embedding`).

</div>
</div>

<div class="keypoint">

A matrix multiply by a one-hot is just a **gather**. The embedding layer is the one "weight matrix" in the network you can read without doing any math — its row $i$ is literally *what token $i$ means to the model.*

</div>

---

# Meaning shows up as directions: `king − man + woman ≈ queen`

Train these vectors on real text and the *geometry* becomes meaningful — differences between vectors encode **relationships**.

<div class="columns">
<div>

**Analogy arithmetic** (word2vec, Mikolov 2013):
$$v_{\text{king}} - v_{\text{man}} + v_{\text{woman}} \approx v_{\text{queen}}$$

The vector $v_{\text{king}}-v_{\text{man}}$ is a "royalty minus male" *direction*; add "female" and you land near **queen**.

</div>
<div>

![w:430px](figures/Lnew/svg/embedding_geometry.svg)

</div>
</div>

<div class="keypoint">

Nobody *told* the model about royalty or capitals. Predicting words from their neighbours (**word2vec / GloVe**) forces words that share contexts into shared directions. Meaning is a **side effect** of next-token prediction.

</div>

---

# The metric of the space: cosine similarity

We compare embeddings by **angle**, not distance — direction carries the meaning, length often just tracks word frequency.

<div class="math-box">

$$\cos(\mathbf u,\mathbf v) = \frac{\mathbf u \cdot \mathbf v}{\lVert\mathbf u\rVert\,\lVert\mathbf v\rVert} \in [-1,1]$$

</div>

- $\cos \to 1$ · same direction → **synonyms / related** ("cat", "dog")
- $\cos \to 0$ · orthogonal → **unrelated** ("cat", "truck")
- $\cos \to -1$ · opposite → **antonyms / contrast**

This is the same dot-product-then-normalize you'll see *scaled* inside attention (L14) — there it scores how much one token should attend to another.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A 2-D embedding table holds

| token | $d_1$ | $d_2$ |
|:-:|:-:|:-:|
| king | 4 | 3 |
| queen | 3 | 4 |
| truck | −3 | 4 |

(a) "Look up" the vectors for **king** and **truck**.
(b) Compute $\cos(\text{king}, \text{queen})$ and $\cos(\text{king}, \text{truck})$.
(c) Which word is the nearest neighbour of **king**, and does the geometry make sense?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

Every row has norm $\sqrt{4^2+3^2}=\sqrt{3^2+4^2}=5$.

$$\cos(\text{king},\text{queen}) = \frac{4\cdot3 + 3\cdot4}{5\cdot5} = \frac{24}{25} = \mathbf{0.96}$$

$$\cos(\text{king},\text{truck}) = \frac{4\cdot(-3) + 3\cdot4}{5\cdot5} = \frac{-12+12}{25} = \mathbf{0}$$

<div class="keypoint">

**queen** is king's nearest neighbour (angle ≈ 16°); **truck** is orthogonal (angle 90°) — unrelated. The lookup was literally reading a row; the meaning is entirely in the *angles*. This is exactly the query–key score attention will compute per pair.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2 · analogy arithmetic.** In a 2-D toy space:

| token | $d_1$ | $d_2$ |
|:-:|:-:|:-:|
| king | 4 | 3 |
| man | 3 | 1 |
| woman | 1 | 2 |
| queen | 2 | 4 |

(a) Compute the "royalty" direction $v_{\text{king}}-v_{\text{man}}$.
(b) Form the analogy vector $v_{\text{king}}-v_{\text{man}}+v_{\text{woman}}$.
(c) Of the four tokens, which is **nearest** to that result — and does `king − man + woman ≈ queen` hold here?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

<div class="columns">
<div>

**(a)** The "royalty minus male" direction:
$$v_{\text{king}}-v_{\text{man}} = (4,3)-(3,1) = (1,2)$$

**(b)** Transport it onto "woman":
$$(1,2) + v_{\text{woman}} = (1,2)+(1,2) = (2,4)$$

</div>
<div>

**(c)** The result $(2,4)$ is *exactly* $v_{\text{queen}}$ — distance $0$. The others are far:
- king $(4,3)$: dist $\sqrt{4+1}=2.24$
- woman $(1,2)$: dist $\sqrt{1+4}=2.24$

So `king − man + woman = queen`. ✓

</div>
</div>

<div class="keypoint">

The analogy is **pure vector arithmetic**: subtract to *isolate* a relationship ("royalty"), add to *transport* it onto a new word ("woman"). Nobody encoded "queen = royal female" — it fell out of *where next-token prediction placed the four vectors*. Real word2vec does this in 300-D, landing *near* queen and reading it off by cosine nearest-neighbour rather than exact equality.

</div>

---

# Build it from scratch: next-character prediction (makemore)

The cleanest place to see embeddings *earn their keep* is a character-level name generator. Frame generation as **classification**: given a 3-character context, predict the next character.

<div class="columns">
<div>

From the name `abid`, with context length 3:

| $X$ (context) | | $Y$ (next) |
|:-:|:-:|:-:|
| `[·, ·, ·]` | → | a |
| `[·, ·, a]` | → | b |
| `[·, a, b]` | → | i |
| `[a, b, i]` | → | d |
| `[b, i, d]` | → | · |

</div>
<div>

- Vocabulary: 26 letters + end marker `·` = **27 classes**
- One name → several $(X,Y)$ training pairs
- The target is a 27-way **categorical** — same softmax + cross-entropy from L1

Character-level keeps the vocabulary tiny; the *mechanism* is identical to a word- or subword-level LLM.

</div>
</div>

---

# Concatenate embeddings → MLP → softmax

<div class="columns">
<div>

**The forward pass** for context `[a, b, i]`:
1. **Look up** each character's row of $E$ ($27 \times K$):
$$a\to[0.2,-0.1],\; b\to[-0.3,0.5],\; i\to[0.1,0.3]$$
2. **Concatenate** → one feature vector
$$[0.2,-0.1,\,-0.3,0.5,\,0.1,0.3] \in \mathbb R^{3K}$$
3. **MLP** → 27 logits → **softmax** → $P(\text{next char})$

</div>
<div>

<div class="math-box">

$$\mathbf h = \tanh(W_1\,[\,E_{x_1};E_{x_2};E_{x_3}\,] + b_1)$$
$$\hat{\mathbf p} = \operatorname{softmax}(W_2 \mathbf h + b_2)$$

</div>

Train with cross-entropy against the true next character. **Two** things learn at once:
- the **embedding table** $E$ ($27\times K$)
- the **MLP weights** $W_1, W_2$

Backprop flows through the softmax, through the MLP, *into the rows of $E$ that were looked up.*

</div>
</div>

---

# What the geometry learns, for free

Because letters that predict similar continuations get similar gradients, the embedding table self-organizes — no labels for "vowel" anywhere.

<div class="columns">
<div>

**A representation emerges:**
- vowels `a e i o u` drift into one cluster
- consonants into another
- the end-marker `·` sits off on its own

</div>
<div>

<div class="insight">

This is **representation learning** in miniature: the same pressure that makes `king−man+woman≈queen` at word scale makes vowels cluster at character scale. Predict-the-next-token *is* the teacher.

</div>

</div>
</div>

The trained table is a $27\times K$ matrix you can inspect, plot in 2-D, and read off similarities — meaning you built with nothing but cross-entropy.

---

# Sampling names: the temperature knob

To *generate*, feed a context, sample the next character from $\hat{\mathbf p}$, slide the window, repeat until `·`. **Temperature** $T$ reshapes the softmax before sampling:

<div class="math-box">

$$P(y_i) = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$$

</div>

<div class="columns">
<div>

- $T \to 0$ · peaked → **greedy**, safe, repetitive
- $T = 1$ · the model's own distribution
- $T \to \infty$ · flat → **uniform**, chaotic

</div>
<div>

| next char | $T{=}1$ | $T{=}0.5$ | $T{=}2$ |
|:-:|:-:|:-:|:-:|
| d | **0.60** | **0.95** | 0.25 |
| s | 0.08 | 0.01 | 0.12 |
| others | 0.32 | 0.04 | 0.63 |

</div>
</div>

Low $T$ → conservative, name-like; high $T$ → creative, sometimes nonsense. We'll meet this exact knob again in **decoding** (Part 5).

---

# See it yourself

<div class="notebook">

**📓 Notebook · makemore from scratch** — build the $27\times K$ embedding table, concatenate-embeddings → MLP → softmax, train on Indian names, then sample with temperature. Inspect the learned 2-D embedding and watch the vowels cluster. *(ML ES 335 · `notebooks/names.ipynb`, after A. Karpathy's "makemore")*

</div>

<div class="notebook">

**🎛 Interactive · softmax temperature** — drag $T$ and watch a next-token distribution sharpen and flatten; see how sampling changes. *(Interactive Lab · `interactive/softmax-temperature`)*

</div>

---

<!-- _class: section-divider -->

## Part 2 · Encoder–decoder — mapping one sequence to another

---

# A new shape of problem

An RNN/LSTM reads a sequence and emits **one label** or a **same-length** stream. Many tasks map a sequence to a *different* sequence — different length, different order:

- translation · 14 English words → ~12 Hindi words, reordered
- summarization · article → abstract
- speech recognition · audio → text

<div class="popquiz">

**Stop and think.** Translate *"The book that the student read was long"* into Hindi. Why does **word-by-word with a dictionary** fail here? *(Hint: where do the words end up?)*

</div>

*Answer:* Hindi reorders the clauses — there is no per-word alignment to follow. We need a model that reads the **whole** source before it commits to any target word.

---

# Encoder → context → decoder, in plain words

Before an equation, the entire mechanism — **two RNNs, back to back:**

1. **Encoder** reads the whole source $(x_1,\dots,x_T)$, updating its hidden state each step.
2. **Context** $\mathbf c$ = the encoder's *final* hidden state — a compressed summary of the source.
3. **Decoder** starts from $\mathbf c$ and writes the target $(y_1, y_2, \dots)$ one token at a time.

<div class="insight">

**Elegant** · no grammar rules, no alignment tables — just two RNNs trained end-to-end on parallel text. **Fragile** · the *entire* sentence, 5 words or 500, must squeeze through that one vector $\mathbf c$. Hold that thought — that squeeze is the **bottleneck** the whole lecture builds toward.

</div>

---

# The architecture

![w:920px](figures/lec11/svg/seq2seq_bottleneck.svg)

<div class="realworld">

Sutskever, Vinyals &amp; Le (2014), *"Sequence to Sequence Learning with Neural Networks"* — this exact recipe hit BLEU 34 on English→French, overtaking hand-engineered phrase-based systems. *(BLEU · n-gram overlap with reference translations, 0–100, higher better.)*

</div>

---

# The whole idea in one sentence

<div class="keypoint">

**Compress the source into a vector · decompress it into the target.**

</div>

A human translator does the same: read the whole German sentence, form a **mental summary**, then compose the English by unpacking it. The encoder builds that summary $\mathbf c$; the decoder unpacks it.

The problem is already visible: for a *long* sentence, even a great translator's single mental snapshot loses detail. So will $\mathbf c$. The same encoder–decoder **pattern** returns in T5, Whisper, and Stable Diffusion — only the *implementation of "context"* changes (fixed vector → attention → self-attention).

---

# Intuition · everything the one vector must remember

The encoder's final state $\mathbf c$ is a **"thought vector"** — Ng's picture: read the whole sentence, form a single mental summary, then speak from it. But list what that *one* vector has to hold for a faithful translation:

<div class="columns">
<div>

- **who did what to whom** — subject, verb, object
- **tense & aspect** — did it happen, will it, is it ongoing
- **number & gender** — one dog or many; agreement downstream
- **tone & negation** — "not happy" must not decompress to "happy"

</div>
<div>

<div class="insight">

For *"The cat sat"* one vector is luxurious. For a 40-word clause with three sub-clauses, the *same* fixed $\mathbf c$ must cram all of the above for *every* phrase — with no more room than it had for three words. **Compress-then-decompress is elegant right up until the thing you compress won't fit.**

</div>

</div>
</div>

The mismatch — *constant* capacity, *growing* demand — is the bottleneck we make precise in Part 4.

---

# Seq2Seq in PyTorch · skeleton

```python
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_emb=256, d_h=512):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_emb)   # the Part-1 table
        self.tgt_emb = nn.Embedding(tgt_vocab, d_emb)
        self.encoder = nn.LSTM(d_emb, d_h, batch_first=True)
        self.decoder = nn.LSTM(d_emb, d_h, batch_first=True)
        self.output  = nn.Linear(d_h, tgt_vocab)

    def forward(self, src, tgt):
        _, (h, c) = self.encoder(self.src_emb(src))     # context = final state
        dec_out, _ = self.decoder(self.tgt_emb(tgt), (h, c))
        return self.output(dec_out)                     # logits over tgt vocab
```

The `nn.Embedding` layers are exactly the learnable lookup tables from Part 1. The encoder keeps only its **final** `(h, c)` — that tuple *is* the context vector we hand the decoder.

---

# One table or two? Shared vs. separate vocabulary

The skeleton gave the encoder and decoder **separate** embedding tables. That is one of two choices:

<div class="columns">
<div>

**Separate** · source = 40k English tokens, target = 40k French tokens, each with its own $E$. Clean; each side can specialize.

</div>
<div>

**Shared** · one vocabulary, one $E$ for both. Fewer parameters; "Paris" is the *same* token in both languages.

</div>
</div>

<div class="realworld">

Modern multilingual models share **one subword vocabulary** across 100+ languages — mT5 / NLLB use SentencePiece, most LLMs use BPE (the tokenization story of L16). One token stream, many languages, one embedding table.

</div>

---

<!-- _class: section-divider -->

## Part 3 · Teacher forcing vs. autoregressive — and exposure bias

---

# The problem with training auto-regressively

At **inference** the decoder feeds its own predictions back in:

> `<s>` → emits "The" → sees "The" → emits "dog" → sees "dog" → …

If we trained the same way and the decoder's first guess is wrong, the error **compounds**: every later step conditions on a corrupted history. Early training — when every guess is wrong — becomes painfully slow and unstable.

<div class="insight">

We want each step to be a *clean* prediction problem, not a hostage to step 1's mistake. The fix is to feed the **ground-truth** previous token during training.

</div>

---

# Teacher forcing · the fix

![w:900px](figures/lec11/svg/teacher_forcing.svg)

Like riding a bike with a parent holding the seat: you still pedal and steer (predict the next word), but when you wobble they keep you upright (feed the true word). You learn the core motion fast; at inference the training wheels come off.

---

# Two regimes, and what teacher forcing optimizes

<div class="columns">
<div>

### Autoregressive (inference)
$$y_{t-1}^{\text{pred}} \to \text{decoder} \to y_t^{\text{pred}}$$
Decoder feeds **its own** prediction back. One error at step 1 → wrong context for *every* later step.

### Teacher forcing (training)
$$y_{t-1}^{\text{truth}} \to \text{decoder} \to y_t^{\text{pred}}$$
Decoder sees the **ground-truth** previous token. Every step is a clean, independent prediction.

</div>
<div>

<div class="math-box">

$$\mathcal L = -\sum_{t=1}^{T}\log p\bigl(y_t \mid y_{<t}^{*},\,x;\,\theta\bigr)$$

</div>

The **conditional NLL** of the target given the *true* history — the L1 next-token loss, summed over the sequence.

**Why it's fast:** inputs `<s>, y*₁, y*₂, …` are *all known up front*, so we drop the per-token sampling loop and score every position in one batched forward/backward — no generate-one-token-at-a-time.

</div>
</div>

---

# Intuition · why the training wheels make it *fast*

Teacher forcing isn't only more stable — it changes the **shape of the computation**. Because every decoder input is known up front, the whole target is one batched tensor.

<div class="columns">
<div>

**Autoregressive training** (naïve)
For a length-$T$ target you must:
sample $y_1 \to$ embed $\to$ feed $\to$ sample $y_2 \to \dots$
$T$ **sequential, data-dependent** steps — you cannot begin step $t$ until step $t-1$ is sampled. Nothing batches across positions.

</div>
<div>

**Teacher forcing**
Inputs $\langle s\rangle, y_1^{*},\dots,y_{T-1}^{*}$ are *all* known, so all $T$ positions go through **one** forward/backward. On a Transformer this is fully parallel across $t$; even on an RNN you drop the per-token *sampling* loop.

</div>
</div>

<div class="keypoint">

Rough count: a $T{=}30$ target trains in **one** batched pass instead of **30** generate-one-token passes. That constant factor is why every large model is trained teacher-forced. The bill for that speed — **exposure bias** — is next.

</div>

---

# Exposure bias · the price you pay

![w:900px](figures/lec11/svg/exposure_bias.svg)

**The cascade.** Source: *"Le chien a chassé le chat."* Target: *"The dog chased the cat."*
At inference the decoder predicts "A" instead of "The", then **feeds "A" back** — but it *never saw its own mistakes in training*, so it cannot recover: "A dog ran away…", fully diverged. It trained in a perfect world and is tested in a messy one. *Mitigations:* scheduled sampling (Bengio 2015), or just scale up (Transformers, L15).

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Teacher forcing feeds the decoder the ground-truth previous token at every step during training.

(a) Why does this make training **much faster** than letting the decoder run on its own predictions?
(b) The model is trained on $p(y_t\mid y_{<t}^{*},x)$ but at inference sees $p(y_t\mid \hat y_{<t},x)$. Name this train/test mismatch, and explain in one line why it causes generation to **derail** after one wrong token.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="columns">
<div>

**(a) Speed.** The decoder inputs `<s>, y*₁, …, y*_{T-1}` are the *known* target, so there is **no sampling loop** — no "predict, embed, feed back" per token. All $T$ positions run through one optimized kernel and are scored in a **single** forward/backward. (The RNN is still sequential in $t$; the win is removing per-step *sampling*.)

</div>
<div>

**(b) Exposure bias.** The model only ever conditions on **correct** history $y_{<t}^{*}$. It is never *exposed* to its own errors, so its behaviour on a wrong prefix is **off-distribution**. At inference, one wrong token puts it on a prefix it never trained on → the next prediction is unreliable → errors compound → the sequence diverges.

</div>
</div>

<div class="keypoint">

The very trick that buys the speedup — always feeding perfect history — is *exactly* what creates the mismatch. Speed and exposure bias are two faces of one coin.

</div>

---

<!-- _class: section-divider -->

## Part 4 · The fixed-context bottleneck — why attention is inevitable

---

# The bottleneck in one sentence

<div class="keypoint">

The entire source — 5 words or 500 — must compress into **one fixed-size context vector** $\mathbf c$. The decoder then generates everything from that single vector.

</div>

For short sentences, fine. For long sentences the encoder **forgets the beginning** by the time it reaches the end, and the decoder has no way to recover what was lost. This is the "fragile" half of the object we met on the architecture slide — now it has a name.

---

# Intuition · one vector can't hold a 40-word sentence

Before the math, the picture. A fixed context vector is a **fixed-size suitcase**. Packing for a weekend is easy; packing for a month into the *same* suitcase means something gets left behind.

<div class="columns">
<div>

**5 words → roomy.** $\mathbf c$ has spare coordinates for every nuance; nothing is lost.

**40 words → overflowing.** The encoder keeps overwriting the same $d$ numbers; by the last word the first clause is a faded memory. The decoder asks "what was the subject?" and $\mathbf c$ no longer knows.

</div>
<div>

<div class="insight">

Ng's framing: a human translator *also* holds one mental summary — but a human can **glance back** at the source sentence. The seq2seq decoder cannot: once the source is compressed into $\mathbf c$, the original words are **gone**. Attention (L14) is precisely "let the decoder glance back."

</div>

</div>
</div>

The next slide turns "won't fit" into a **counting argument** — demand grows with length, supply is fixed.

---

# Why it *must* fail: a capacity argument

Treat $\mathbf c$ as a communication channel with **fixed** capacity $B$ bits (a $d$-dim vector holds only so much usable information — finite precision, finite $d$).

<div class="columns">
<div>

**Demand grows with length.** A sentence of $n$ tokens over vocabulary $V$ carries up to
$$n\log_2 V \text{ bits}$$
Information *needed* rises **linearly** in $n$.

</div>
<div>

**Supply is constant.** The context can deliver at most $B$ bits, whatever $n$ is.

$$\text{once } n\log_2 V > B \;\Rightarrow\; \text{loss is forced}$$

</div>
</div>

<div class="math-box">

$$n^{\star} \approx \frac{B}{\log_2 V} \quad\Longrightarrow\quad \text{beyond } n^{\star}, \text{ distinct sentences must collide in } \mathbf c.$$

</div>

No training trick removes this: a line (demand) eventually crosses a horizontal line (supply). The decoder receives an ambiguous $\mathbf c$ and *cannot* disambiguate.

---

# Worked numbers · where does the cliff fall?

Put numbers on $n^{\star}\approx B/\log_2 V$. Take a $d=512$ context vector and a $V=32{,}000$ subword vocabulary.

<div class="columns">
<div>

**Vocabulary term.** $\log_2 32{,}000 \approx 15$ bits — the information in *naming one token* out of the vocabulary.

**Channel term.** Suppose each of the 512 coordinates reliably carries $\approx 4$ usable bits (finite precision, noise):
$$B \approx 512 \times 4 = 2048 \text{ bits}$$

</div>
<div>

**The crossover.**
$$n^{\star} \approx \frac{2048}{15} \approx \mathbf{136}\ \textbf{tokens}$$

Below ~130 tokens $\mathbf c$ *can*, in principle, keep every sentence distinct; past it, distinct sentences **must** collide. Real systems crack far earlier (~20–40 words) — coordinates are nowhere near 4 clean bits and the encoder wastes capacity — but the *shape* is exactly this.

</div>
</div>

<div class="keypoint">

The number is a back-of-envelope, not a spec — the point is the **structure**: demand that rises with $n$ against a supply that doesn't. Change $B$ or $V$ and the cliff *moves*; it never disappears.

</div>

---

# The symptom: BLEU collapses as sentences grow

Plot translation quality against source length and you see a **cliff**, not a gentle slope — the signature of a fixed channel saturating.

| source length | quality (BLEU) | what's happening |
|:-:|:-:|:--|
| ≤ 10 tokens | high | $\mathbf c$ holds the whole sentence comfortably |
| ~20 tokens | slipping | early words fading from $\mathbf c$ |
| 40+ tokens | collapse | demand $\gg$ capacity; $\mathbf c$ is saturated |

<div class="insight">

Short sentences hide the flaw entirely — which is why the 2014 results looked great on average and *still* pointed straight at the problem. The average masks the cliff.

</div>

---

# A hack that reveals the disease: reverse the input

Sutskever's own paper found that **reversing the source** gained several BLEU points:

> `"I am happy"` → encode in order
> `"happy am I"` → encode **reversed**

Why would that help? Reversing puts the *first* source words nearest the decoder's start — shortening the path their information travels through the hidden-state chain.

<div class="warning">

**A hack that works is a signal.** Reversing didn't *remove* the bottleneck — it just moved *where* the leakage happens. When a trick this arbitrary buys real gains, the architecture has a hole waiting to be filled properly.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4.** A seq2seq model uses the encoder's final hidden state $\mathbf c \in \mathbb R^{512}$ as its only context.

(a) Model $\mathbf c$ as a channel of fixed capacity $B$ bits. For a sentence of $n$ tokens over vocabulary $V$, write the information the decoder *needs* vs. the information $\mathbf c$ can *supply*, and argue there is a length $n^{\star}$ beyond which quality **must** degrade.
(b) A 50-word sentence's subject-verb agreement depends on word #1. Why is "final hidden state" the *worst* place to store it?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="columns">
<div>

**(a)** Needed information grows **linearly**: $\;n\log_2 V$ bits. Supplied information is **constant**: $\;B$ bits. Setting demand = supply,
$$n^{\star} \approx \frac{B}{\log_2 V}.$$
For $n>n^{\star}$ the map "sentence → $\mathbf c$" cannot be injective: distinct sentences **collide** in the same $\mathbf c$, so the decoder cannot recover them → quality degrades. A rising line always crosses a flat one.

</div>
<div>

**(b)** In an RNN, $\mathbf c=h_T$ is dominated by *recent* inputs; word #1's contribution has been overwritten across ~50 nonlinear updates (the vanishing-signal problem). Storing a long-range dependency in the **last** state is maximally fragile — it's the information most likely to have decayed.

</div>
</div>

<div class="keypoint">

Both parts point the same way: **stop forcing everything through one vector.** Let the decoder read *all* encoder states and choose what to look at — that is attention.

</div>

---

# See it yourself · watch the vector saturate

<div class="notebook">

**🎛 Interactive · the seq2seq bottleneck** — push sentences of growing length through a fixed-size context vector and watch reconstruction quality fall off a cliff as demand crosses capacity. Drag the context dimension $d$ up and see the cliff slide *right* — never vanish. *(Interactive Lab · `interactive/seq2seq-bottleneck`)*

</div>

<div class="notebook">

**📓 Notebook · a minimal encoder–decoder** — the `Seq2Seq` skeleton from Part 2, trained on a toy copy/reverse task; plot BLEU vs. source length and reproduce the cliff yourself, *before* attention (L14) erases it. *(ES 667 Seq2Seq materials.)*

</div>

<div class="popquiz">

**Predict first.** Before you drag $d$: doubling the context dimension roughly **doubles** $B$. By the counting argument, what happens to the crossover length $n^{\star}$ — and does that *solve* the bottleneck or merely postpone it?

</div>

---

# The obvious next step → attention (L14)

If one context vector can't hold all the source information, **don't use one context vector.**

Instead, keep *every* encoder hidden state, and let the decoder — at each target step — **look back** and decide which source positions matter *right now*.

<div class="keypoint">

The bottleneck we teased on the very first architecture slide is *exactly* what **attention** removes — not by shifting the leak (like reversing the input) but by deleting the single-vector squeeze entirely. **Bahdanau et al. (2014)** launched a decade of NLP with this idea. That is **L14**.

</div>

---

<!-- _class: section-divider -->

## Part 5 · Decoding — from probabilities to a sentence

---

# The model gives probabilities — you still must pick words

At each step the decoder emits a distribution over the vocabulary. Turning that into text is a **search** problem with three standard answers:

![w:600px](figures/lec11/svg/decoding_strategies.svg)

- **Greedy** · take $\arg\max$ every step — fast, deterministic, often globally suboptimal
- **Beam** · keep the top-$k$ partial sentences alive — better sequences, $k\times$ cost
- **Sampling** · draw from the distribution (with temperature / top-$k$ / nucleus) — diverse, creative

---

# Greedy is short-sighted; beam keeps $k$ drafts alive

<div class="columns">
<div>

**Greedy fails** — locally best ≠ globally best. If step 1 offers "A" (0.6) vs "The" (0.4):
- best path from "A": $0.6\times0.3=0.18$
- best path from "The": $0.4\times0.8=\mathbf{0.32}$

Greedy commits to "A" and never reaches the better sentence.

</div>
<div>

**Beam search** keeps the $k$ best prefixes at every step, scoring by summed log-prob:
$$\text{score}(y)=\sum_{t}\log P(y_t\mid y_{<t})$$

Log-probs are all negative, so long sentences score lower → always preferred *shorter*. Fix with **length normalization**, dividing by $T^{\alpha}$ ($\alpha\approx0.6$–$0.7$).

</div>
</div>

<div class="insight">

Bad output? Ask which error: **search error** (a better sequence exists, beam missed it — raise $k$) or **model error** (the model's own top choice is garbage — no $k$ can fix that).

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5 · beam vs. length bias.** A decoder scores sequences by summed log-probability. Two candidate translations:
- **A** (3 tokens): log-probs $-0.2,\,-0.3,\,-0.5$
- **B** (5 tokens): log-probs $-0.2,\,-0.3,\,-0.4,\,-0.2,\,-0.3$

(a) Which wins on **raw** summed log-prob?
(b) Apply **length normalization** — divide each score by $T^{\alpha}$ with $\alpha=0.7$. Which wins now?
(c) In one line: why does raw beam search systematically prefer *shorter* outputs?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

<div class="columns">
<div>

**(a) Raw sums.**
$$\text{A}=-1.0,\qquad \text{B}=-1.4$$
**A wins** — but only because it has fewer (all-negative) terms to add.

**(b) Length-normalized** ($\div\,T^{0.7}$):
$$\text{A}: \frac{-1.0}{3^{0.7}}=\frac{-1.0}{2.16}=\mathbf{-0.46}$$
$$\text{B}: \frac{-1.4}{5^{0.7}}=\frac{-1.4}{3.09}=\mathbf{-0.45}$$
**B now edges ahead** — its per-token confidence is actually higher.

</div>
<div>

**(c) The length bias.** Every extra token adds another *negative* $\log P$, so a longer sequence's score can only go **down**. Raw beam search therefore rewards brevity for a reason that has nothing to do with quality. Dividing by $T^{\alpha}$ compares sequences on **average** log-prob per token, cancelling the length penalty.

</div>
</div>

<div class="keypoint">

$\alpha$ tunes the correction: $\alpha=0$ is raw (short-biased), $\alpha=1$ is a pure per-token average (can over-reward length). Production decoders land at $\alpha\approx 0.6$–$0.7$ — the value quoted on the beam-search slide.

</div>

---

# Sampling: when you *want* variety

Greedy and beam are deterministic — great for translation, dull for open-ended text (every story reads the same). **Sampling** draws from the distribution, and three knobs shape the pool:

<div class="columns">
<div>

- **Temperature** $T$ · rescale logits by $1/T$ before softmax (Part 1's knob) — low = safe, high = wild
- **Top-$k$** · sample only from the $k$ most likely tokens
- **Top-$p$ (nucleus)** · smallest set with cumulative prob $\ge p$ — **adapts**: narrow when confident, wide when unsure

</div>
<div>

<div class="realworld">

**Current LLM default** — nucleus sampling with $p\approx0.9$–$0.95$ plus a temperature knob. The exact same $e^{z/T}$ softmax you used to sample names in Part 1, now sampling tokens from a Transformer.

</div>

</div>
</div>

---

# Worked example · softmax at two temperatures

Three tokens with logits $z = (2.0,\,1.0,\,0.0)$. Apply $P(y_i)\propto e^{z_i / T}$ at $T=1$ and $T=0.5$.

<div class="columns">
<div>

**$T=1$** — exponentiate the logits:
$$e^{2},e^{1},e^{0} = 7.39,\,2.72,\,1.00$$
sum $=11.11$, so
$$P \approx (0.67,\,0.24,\,0.09)$$

</div>
<div>

**$T=0.5$** — logits double ($z/T = 4,2,0$):
$$e^{4},e^{2},e^{0} = 54.6,\,7.39,\,1.00$$
sum $=63.0$, so
$$P \approx (\mathbf{0.87},\,0.12,\,0.02)$$

</div>
</div>

<div class="keypoint">

Lower $T$ **sharpens** the distribution — the top token's mass climbs $0.67 \to 0.87$, the tail is starved. $T\to 0$ collapses to greedy $\arg\max$; $T\to\infty$ flattens to uniform. It is *the same softmax* from L1 — temperature only rescales the logits **before** it, so it never reorders the tokens, only how peaky the winner is.

</div>

<div class="notebook">

**🎛 Interactive · softmax temperature** — drag $T$ and watch this exact distribution sharpen and flatten, with a live sample. *(Interactive Lab · `interactive/softmax-temperature`)*

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Learn a vector per token so meaning lives in geometry; wire two RNNs into an encoder–decoder — and the single context vector between them is the wall that makes attention inevitable.**

</div>

- **Embeddings** — one-hot is blind to similarity; a learned table $E$ puts meaning in *angles* (`king−man+woman≈queen`), built by nothing but next-token cross-entropy.
- **Seq2Seq** — encoder squeezes the source into $\mathbf c$; decoder unspools $\mathbf c$ into the target.
- **Teacher forcing** trains on true history (fast); **exposure bias** is the price.
- **The bottleneck** — one fixed vector; demand grows linearly with length, capacity is constant → BLEU cliffs on long sentences.
- **Decoding** — greedy / beam (length-normalized) / sampling (temperature, nucleus).

**Next (L14):** stop compressing — let the decoder look back at every source word. **Attention.**

---

<!-- _class: compact -->

# Sources &amp; credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Next-token classification, the embedding table, temperature sampling** — *Machine Learning (ES 335)*, N. Batra · `github.com/nipunbatra/ml-teaching` (`notebooks/names.ipynb`), after **A. Karpathy**, *"makemore" / Neural Networks: Zero to Hero.*
- **Encoder–decoder, teacher forcing, exposure bias, the bottleneck, decoding** — *ES 667 Seq2Seq materials*, N. Batra.
- **Pedagogical framing &amp; sequencing** — A. Ng, *Deep Learning Specialization*, Course 5 (Sequence Models), Weeks 2–3.
- **Word embeddings / analogy arithmetic** — Mikolov et al., *"Efficient Estimation of Word Representations in Vector Space"* &amp; *word2vec* (2013); Pennington et al., *GloVe* (2014).
- **Seq2Seq** — Sutskever, Vinyals &amp; Le, *"Sequence to Sequence Learning with Neural Networks"* (2014). **Attention (next lecture)** — Bahdanau et al. (2014).

Interactive Lab · `interactive/seq2seq-bottleneck`, `interactive/softmax-temperature`. Figures adapted from the ES 667 figure library. All source courses © N. Batra &amp; teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · what word2vec actually optimizes

**Skip-gram** turns "predict-the-context" into a classification loss. For a center word $w$ and a context word $c$, model the probability they co-occur with dot-product embeddings:

$$P(c\mid w) = \frac{\exp(\mathbf u_c^\top \mathbf v_w)}{\sum_{c'\in V}\exp(\mathbf u_{c'}^\top \mathbf v_w)}$$

The denominator sums over the **whole vocabulary** — hopeless at $V=10^6$. **Negative sampling** replaces it with a handful of "is this pair real?" logistic problems:

$$\log\sigma(\mathbf u_c^\top\mathbf v_w) + \sum_{i=1}^{k}\mathbb E_{c_i\sim P_n}\big[\log\sigma(-\mathbf u_{c_i}^\top\mathbf v_w)\big]$$

Push real (word, context) pairs together, random pairs apart. The dot product $\mathbf u_c^\top\mathbf v_w$ is why **angles** end up meaningful — and it is the same score, *scaled by* $1/\sqrt{d_k}$, at the heart of attention (L14). *(word2vec, Mikolov 2013.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · the bottleneck as a rate–distortion limit

Make the capacity argument precise. Let the source be a random sentence $S$ with entropy $H(S)$ bits, encoded into $\mathbf c$, then decoded to $\hat S$. By the **data-processing inequality**,

$$I(S;\hat S) \le I(S;\mathbf c) \le C(\mathbf c),$$

where $C(\mathbf c)$ is the channel capacity of the fixed vector (bounded by its dimension and effective precision). If reconstruction demands mutual information $I(S;\hat S)\ge R(D)$ to hit distortion $D$ (the rate–distortion function), then

$$H(S)\ \uparrow\ \text{with length}, \qquad C(\mathbf c)\ \text{fixed} \;\Longrightarrow\; \text{achievable distortion } D \uparrow.$$

Longer sentences raise $H(S)$ while $C(\mathbf c)$ stays put, so the **best possible** reconstruction error grows — a limit no optimizer can beat. Attention sidesteps it by making the channel's width **scale with the source** (one vector *per* position), so $C$ grows with length instead of staying constant. *(Full treatment: Cover &amp; Thomas, Ch. 10.)*
