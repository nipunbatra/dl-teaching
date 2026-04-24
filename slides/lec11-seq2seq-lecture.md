---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Seq2Seq &amp; the Motivation for Attention

## Lecture 11 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Describe **encoder-decoder** RNN architecture for variable-length tasks.
2. Implement **teacher forcing** and explain exposure bias.
3. Contrast **greedy / beam / top-k / nucleus** decoding.
4. Apply **length normalization** in beam search.
5. Diagnose the **fixed-length bottleneck** that killed pre-attention Seq2Seq.
6. Motivate **attention** (L12) as the fix for the bottleneck.

---

# Recap · where we are

Last lecture: **LSTMs** solve the vanishing-gradient problem in RNNs via gated cell states.

But solving depth is not enough. Many tasks need to map **an input sequence** to **a different output sequence**:

- Machine translation — English to French
- Summarization — article to abstract
- Speech recognition — audio to text
- Program synthesis — comment to code

<div class="paper">

Today maps to **Bishop Ch 12** (Seq2Seq). UDL treats Transformers directly; we cover the encoder-decoder precursor because it *motivates* attention.

</div>

---

# Four questions

1. How do we map a variable-length input to a variable-length output?
2. What is **teacher forcing** and why does everyone use it?
3. How do we **decode** at inference — greedy, beam, nucleus?
4. What's wrong with Seq2Seq — and why did attention become inevitable?

---

<!-- _class: section-divider -->

### PART 1

# The encoder-decoder architecture

Read all, then generate

---

# Seq2Seq · the 2014 breakthrough

<div class="paper">

Sutskever, Vinyals, Le 2014 · *"Sequence to Sequence Learning with Neural Networks"* — achieved BLEU 34 on English→French, within striking distance of phrase-based MT.

</div>

Two separate RNNs:

1. **Encoder** · reads the source sequence $(x_1, \ldots, x_{T})$, updates its hidden state.
2. **Context vector** $\mathbf{c}$ · encoder's final hidden state — a compressed summary of the whole source.
3. **Decoder** · starts from $\mathbf{c}$, generates target tokens $(y_1, y_2, \ldots)$ one at a time.

---

# The whole idea in one sentence

**Compress source into a vector · decompress into target.**

<div class="keypoint">

Two unrolled RNNs, back to back, trained end-to-end. No grammar rules, no alignment dictionaries, no phrase tables — the representations are learned from parallel corpus data alone. This was radically new in 2014; by 2016 it was state-of-the-art in production MT.

</div>

The same encoder-decoder pattern returns in T5 (L14), Stable Diffusion (L22), and every modular ML system that maps between domains.

---

# Shared vs separate vocabularies

Two design choices:

- **Separate vocab** · source is 40k English tokens, target is 40k French tokens, each with its own embedding matrix. Clean; embeddings can specialize.
- **Shared vocab** · one vocabulary for both, one embedding matrix. Saves parameters; lets the model see "Paris" as the same token in both languages.

<div class="realworld">

Modern multilingual models (mT5, NLLB, Whisper) share vocab via SentencePiece — a single token stream covers 100+ languages. Today's LLMs do the same.

</div>

---

# The architecture

![w:920px](figures/lec11/svg/seq2seq_bottleneck.svg)

<div class="realworld">

▶ Interactive: see BLEU curves fall as source length grows — [seq2seq-bottleneck](https://nipunbatra.github.io/interactive-articles/seq2seq-bottleneck/).

</div>

---

# Seq2Seq in PyTorch · skeleton

```python
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_emb=256, d_h=512):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, d_emb)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_emb)
        self.encoder = nn.LSTM(d_emb, d_h, batch_first=True)
        self.decoder = nn.LSTM(d_emb, d_h, batch_first=True)
        self.output  = nn.Linear(d_h, tgt_vocab)

    def forward(self, src, tgt):
        _, (h, c) = self.encoder(self.src_emb(src))     # context = (h, c)
        dec_out, _ = self.decoder(self.tgt_emb(tgt), (h, c))
        return self.output(dec_out)
```

---

<!-- _class: section-divider -->

### PART 2

# Teacher forcing

How we train sequence generators

---

# The problem with training auto-regressively

At **inference** the decoder feeds its own predictions back in:

> decoder sees "\<start\>" → emits "The" → sees "The" → emits "animal" → …

But at **training**, if the decoder's first prediction is wrong, the error compounds — all subsequent steps condition on bad inputs. Training becomes painfully slow and unstable.

---

# Teacher forcing · the fix

![w:920px](figures/lec11/svg/teacher_forcing.svg)

---

# Teacher forcing · detailed flow

![w:920px](figures/lec11/svg/teacher_forcing_detail.svg)

---

# The two training regimes side-by-side

During training, **feed the ground-truth previous token** as the decoder's input, not its own previous prediction.

<div class="columns">
<div>

### Autoregressive (inference)

$$y_{t-1}^{\text{pred}} \to \text{decoder} \to y_t^{\text{pred}}$$

</div>
<div>

### Teacher forcing (training)

$$y_{t-1}^{\text{truth}} \to \text{decoder} \to y_t^{\text{pred}}$$

</div>
</div>

Effect: every target step is trained independently given the correct history. **Massive speedup** and more stable gradients.

---

# Teacher forcing · why it's OK pragmatically

Even though training ≠ inference, teacher forcing works because:

1. **Loss is averaged over all time steps** · bad predictions at step 10 aren't penalized harder than bad predictions at step 1. Every position gets balanced gradient.
2. **The model learns conditional distributions $P(y_t | y_{<t})$** · if you give it the right $y_{<t}$ at inference (as in a perfect rollout), it generalizes.
3. **Huge speedup from parallelization** · with ground truth, all decoder steps can be computed in parallel (one big matrix multiply) instead of sequentially.

<div class="insight">

The "you're training on a distribution you won't see at inference" critique is real (exposure bias, next slide). But empirically it's tolerated because the alternative — fully autoregressive training — is 10-100× slower.

</div>

---

# Exposure bias · the price you pay

![w:900px](figures/lec11/svg/exposure_bias.svg)

---

# Exposure bias · explained

<div class="warning">

Teacher forcing means the model is never exposed during training to *its own* mistakes. At inference, one wrong prediction cascades — the model has never seen that recovery path.

</div>

Mitigations:

- **Scheduled sampling** (Bengio et al. 2015) — probabilistically feed the model's own prediction during training.
- **Data augmentation** with noisy inputs.
- **2020+** — Transformers basically sidestep this through massive scale + mixing.

---

<!-- _class: section-divider -->

### PART 3

# Decoding strategies

How to generate at inference

---

# Four common strategies

![w:920px](figures/lec11/svg/decoding_strategies.svg)

---

# Decoding · the search tree

![w:900px](figures/lec11/svg/decoding_tree.svg)

---

# Greedy · pick top-1 every step

Simplest: at each step, pick $\arg\max_y P(y \mid \text{history})$.

**Fast.** Deterministic. Usually **suboptimal** — a slightly-less-likely next token can lead to a much-more-likely full sequence.

Example (simplified):

> greedy: "The dog is running" (but doesn't quite fit context)
> better: "The puppies are running" (total prob higher)

---

# Greedy fails · the canonical example

Imagine the true best translation is "The cat sits on the mat" (probability 0.7).

At step 1, probabilities are:
- "The" · 0.6 (leads to the correct sequence)
- "A" · 0.8 (leads to "A feline rests...", probability 0.5)

Greedy picks "A" because it's locally higher. But the full-sequence score is lower than starting with "The".

<div class="warning">

**Local optima are not global optima.** Greedy decoding is a greedy search on the product of conditional probabilities — it commits at every step. Beam search (next) mitigates by keeping multiple candidates alive until the sequence ends.

</div>

---

# Beam search · the tree

![w:880px](figures/lec11/svg/beam_search_tree.svg)

---

# Beam search · keep top-$k$ paths

At each step, maintain $k$ candidate prefixes. Expand each by all possible next tokens, keep the top $k$ by joint probability.

<div class="math-box">

**Score** · $\log P(y_1, \ldots, y_T) = \sum_t \log P(y_t \mid y_{<t})$

**Length-normalized** · divide by $T^\alpha$ (typically $\alpha = 0.6$) to avoid preferring short outputs.

</div>

Typical $k = 4$ to $10$. More = better quality, slower. Still deterministic given $k$.

---

# Top-$k$ and nucleus (top-$p$) sampling

For open-ended generation (story writing, chat) beam search is *too* deterministic — everything sounds the same.

- **Top-$k$** — truncate to the $k$ most likely tokens, renormalize, sample.
- **Top-$p$ (nucleus)** — truncate to the smallest set whose cumulative probability $\geq p$, renormalize, sample.

<div class="realworld">

**2026 LLM default** — nucleus with $p = 0.9$ or $p = 0.95$, plus a **temperature** knob (see the softmax-temperature interactive).

</div>

---

<!-- _class: section-divider -->

### PART 4

# The bottleneck that killed Seq2Seq

Why attention became inevitable

---

# The failure mode · source length

![w:920px](figures/lec11/svg/seq2seq_bottleneck.svg)

---

# The bottleneck in one sentence

<div class="keypoint">

The entire source — 5 words or 500 — must compress into one fixed-size context vector. The decoder then generates from that single vector.

</div>

For short sentences, fine. For long sentences, the encoder **forgets** the beginning by the time it reaches the end. The decoder has no way to recover what was lost.

---

# Sutskever's own fix · reverse the input

The 2014 paper itself found that **reversing the source** improved BLEU by several points:

> `"I am happy"` → encoded in order
> `"happy am I"` → encoded *reversed*

Why? The last source words (now first) are closest to where the decoder begins generating — less path length for that information to travel through the hidden-state chain.

<div class="insight">

A hack that works is a sign of a problem waiting to be solved properly. Reversing the input "fixed" the bottleneck by shifting where the leakage happens, not by removing it.

</div>

---

# The obvious next step

If one context vector can't hold all source info, **don't use one context vector**.

Instead, let the decoder *look at* all the encoder hidden states — and decide which ones to focus on for each target step.

<div class="keypoint">

That is **attention**. Bahdanau et al. 2014 — the paper that launched a decade of NLP. Next lecture.

</div>

---

<!-- _class: section-divider -->

### PART 5

# Applications of classic Seq2Seq

Even in 2026, some parts survive

---

# Where Seq2Seq-like ideas still ship

- **Speech recognition** · Whisper uses encoder-decoder with attention (the full recipe).
- **Machine translation** · Transformer encoder-decoder is *literally* Seq2Seq with attention instead of a context vector.
- **Code generation** · same family, different data.
- **Summarization** · T5, BART — encoder-decoder Transformers.

<div class="realworld">

The Seq2Seq **pattern** (encoder → context → decoder) is everywhere. Only the *implementation* of "context" changed: fixed vector (2014) → attention (2015) → self-attention (2017) → ... → your favorite 2026 LLM.

</div>

---

<!-- _class: summary-slide -->

# Lecture 11 — summary

- **Seq2Seq** · encoder-decoder for variable-length input → variable-length output.
- **Context vector** is the encoder's final hidden state — a compressed summary.
- **Teacher forcing** trains the decoder with ground-truth history; **exposure bias** is the price.
- **Decoding** — greedy, beam (length-normalized), top-k, **nucleus (2026 default)**.
- **The bottleneck** — one fixed vector for all source info; BLEU collapses past ~30 tokens.
- **Attention** is the fix · next lecture opens Module 6.

### Read before Lecture 12

**Prince Ch 12** (early sections on attention and QKV).

### Next lecture

**The Attention Mechanism** — Bahdanau additive · Luong multiplicative · the query-key-value abstraction · scaled dot-product and why $\sqrt{d_k}$ · self-attention.

<div class="notebook">

**Notebook 11** · `11-seq2seq-nmt.ipynb` — tiny English→French translator with an LSTM encoder-decoder; no attention yet. Next notebook adds attention and you'll see the BLEU gap close.

</div>
