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

Last lecture (L10): **LSTM gates** fixed the vanishing gradient, so a single RNN can now read a long sequence and emit one label or one continuation.

But that's still **one sequence in, same-length stream out**. Many tasks map **an input sequence** to a *different* output sequence — different length, different order:

- Machine translation — English to French
- Summarization — article to abstract
- Speech recognition — audio to text
- Program synthesis — comment to code

<div class="insight">

L10 made the RNN *remember*. Today we wire two of them — encoder + decoder — and hit a new wall · the encoder must cram the whole source into **one fixed vector**. That bottleneck is today's flagship failure, and it's exactly what makes attention (L12) inevitable.

</div>

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

# Pop quiz · the translation challenge

You're translating ·

> *"The book that the student who lived in Paris read was very long."*

into Hindi · 14 English words → ~12 Hindi words in *different* order.

<div class="popquiz">

(a) Word-by-word with a dictionary.
(b) An RNN that reads the whole sentence into a single fixed vector, then writes Hindi.
(c) A model that can **look back** at any English word while writing each Hindi word.

Stop and think · why does (a) fail? Why does (b) struggle on long sentences? (c) is attention — but we'll only get there in L12. Today's lecture is *(b) + its bottleneck*.

</div>

---

▶ **Interactive** · watch the encoder bottleneck struggle as sentences grow → [seq2seq-bottleneck](https://nipunbatra.github.io/interactive-articles/seq2seq-bottleneck/).

---

<!-- _class: section-divider -->

### PART 1

# The encoder-decoder architecture

How do you map 14 words to 12 — in a different order?

---

# Seq2Seq · the 2014 breakthrough

<div class="paper">

Sutskever, Vinyals, Le 2014 · *"Sequence to Sequence Learning with Neural Networks"* — achieved BLEU 34 on English→French, within striking distance of phrase-based MT.

*(BLEU · n-gram overlap between system output and reference translations, 0–100, higher better. The standard MT score.)*

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

# The translator analogy

<div class="keypoint">

A human translator faced with a long German sentence does not translate word-by-word. They **read the whole sentence**, pause to grasp the meaning, **form a mental summary**, then start composing the English translation by unpacking that summary.

Seq2Seq does exactly this · the encoder reads, builds a context vector (the "mental summary"), and the decoder unpacks it into the target language.

</div>

The problem · for *long* sentences, even a great human's mental summary fails. So does Seq2Seq · this is why we'll add attention in L12.

---

# The architecture

![w:920px](figures/lec11/svg/seq2seq_bottleneck.svg)

<div class="realworld">

▶ Interactive: see BLEU curves fall as source length grows — [seq2seq-bottleneck](https://nipunbatra.github.io/interactive-articles/seq2seq-bottleneck/).

</div>

---

# A quick look inside `nn.LSTM`

An `nn.LSTM` layer is a function with specific I/O:

- **Input** · sequence of embedded tokens (e.g. 10-word sentence → `(10, 256)`).
- **Output** · returns **two** things:
  1. `outputs` — hidden state at *every* time step (e.g. `(10, 512)`).
  2. `(h_n, c_n)` — *final* hidden + cell state (each `(1, 512)`).

For the encoder we want the **final** state — that's our context vector $\mathbf{c}$. We discard `outputs` (using `_` in Python) and keep the tuple `(h, c)`.

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

Should training feed the model its own mistakes?

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

# Teacher forcing · the training-wheels analogy

<div class="insight">

Like learning to ride a bike with a parent holding the seat. You still pedal and steer (predict the next word). But when you wobble (make a mistake), the parent keeps you on the right path (feeds you the ground-truth word).

You learn the core motion much faster and more safely. At inference, the training wheels come off.

</div>

---

# Two regimes side-by-side

<div class="columns">
<div>

### Autoregressive (inference)

$$y_{t-1}^{\text{pred}} \to \text{decoder} \to y_t^{\text{pred}}$$

The decoder feeds **its own** prediction back in. One error at step 1 → wrong context for *every* later step.

</div>
<div>

### Teacher forcing (training)

$$y_{t-1}^{\text{truth}} \to \text{decoder} \to y_t^{\text{pred}}$$

The decoder sees the **ground-truth** previous token. Every step is a clean, independent prediction problem.

</div>
</div>

---

# Why teacher forcing? · speed

The biggest reason for teacher forcing isn't safety — it's **parallelism**.

- **Autoregressive** · compute step 1 → step 2 → step 3 → … sequential.
- **Teacher forcing** · all decoder inputs `<s>, "The", "cat", …` are known up-front → feed them in **all at once** → one big matrix multiply.

A slow sequential loop becomes a fast parallel computation — **10–100× speedup** on training. (The unrolled graph still stores activations for every timestep, so memory grows with sequence length either way.)

---

# Teacher forcing · what it actually optimizes

<div class="math-box">

$$\mathcal{L} = -\sum_{t=1}^{T} \log p\bigl(y_t \mid y_{<t}^{*},\, x;\, \theta\bigr)$$

The **conditional likelihood** of the target given the *true* history — the sequence version of the L00 NLL story.

</div>

Why this is statistically OK:

1. **Loss is averaged over all time steps** · every position gets balanced gradient.
2. **The model learns the right conditionals $P(y_t \mid y_{<t})$** · given the correct history at inference, it generalizes.

The "you train on a distribution you won't see at inference" critique is real — that's **exposure bias**, next slide.

---

# Exposure bias · the price you pay

![w:900px](figures/lec11/svg/exposure_bias.svg)

---

# Exposure bias · concrete cascade

Source: "Le chien a chassé le chat." Target: "The dog chased the cat."

**Training (teacher forcing).**
1. Input `<s>` → predicts "A" (logP −1.2). Truth is "The". Loss is computed.
2. Next input is the ground truth `"The"`. **Back on track.**

**Inference (autoregressive).**
1. Input `<s>` → predicts "A".
2. Next input is **"A"** (its own mistake).
3. Conditioned on "A", the model now predicts "dog" (probable continuation of "A").
4. Sequence so far · "A dog…". The model has **never seen** its own mistakes during training, so it has no idea how to recover. Continues "A dog ran away" — completely diverged.

The model trained in a perfect world, tested in a messy one.

**Mitigations** · scheduled sampling (Bengio 2015), noisy data augmentation, or just sidestep with massive-scale Transformers (2020+).

---

<!-- _class: section-divider -->

### PART 3

# Decoding strategies

The model outputs probabilities — how do you pick the words?

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

At step 1, the model's next-token probabilities are:
- "A" · 0.6 — locally the best
- "The" · 0.4

Greedy commits to "A". But follow each path to the end:
- Best sequence starting "A" · "A feline rests…" · $0.6 \times 0.3 = \mathbf{0.18}$
- Best sequence starting "The" · "The cat sits on the mat" · $0.4 \times 0.8 = \mathbf{0.32}$

The globally best translation began with the *locally worse* token.

<div class="warning">

**Local optima are not global optima.** Greedy decoding is a greedy search on the product of conditional probabilities — it commits at every step. Beam search (next) mitigates by keeping multiple candidates alive until the sequence ends.

</div>

---

# Beam search · the tree

![w:880px](figures/lec11/svg/beam_search_tree.svg)

---

# Beam search · the team-of-hikers analogy

<div class="keypoint">

Greedy decoding is **one hiker** taking the steepest step at every fork. They miss bigger peaks that require a less steep early path.

Beam search sends out **$k$ hikers** (the "beam"). At every fork, they explore in parallel. We rank all paths globally and keep the $k$ most-promising survivors. Repeat.

</div>

The team's collective best end-of-path score is far closer to the global maximum than greedy's. Cost · $k\times$ compute. Quality · 2-5 BLEU points typically.

---

# From probabilities to log-probabilities

We want the sentence with highest $P(y_1, \ldots, y_T)$:
$P = P(y_1)\,P(y_2|y_1)\,\cdots$

For a 20-word sentence we'd multiply 20 small probabilities → **numerical underflow** (rounded to 0).

**Fix · take logs.** $\log(ab) = \log a + \log b$. Multiplying probabilities becomes **adding log-probabilities** — much more stable.

$$\text{score}(y) = \sum_{t=1}^T \log P(y_t \mid y_{<t})$$

Each $\log P$ is a *negative* number. Higher (less negative) = better.

---

# The length-bias problem

Every $\log P$ is negative → longer sentences accumulate **more** negatives → lower scores → **always preferred shorter**.

| Sentence | logP |
|---|---|
| "I am" | $-0.5 + (-0.8) = -1.3$ |
| "I am a student" | $-0.5 + (-0.8) + (-0.6) + (-0.4) = -2.3$ |

Plain log-prob picks "I am". Wrong.

**Length normalization** · divide by sentence length raised to $\alpha$:
$$\text{score}(y) = \frac{1}{T^\alpha}\sum_{t=1}^T \log P(y_t \mid y_{<t})$$

$\alpha = 0$ · no penalty (broken) · $\alpha = 1$ · per-token average · **$\alpha \approx 0.6$–$0.7$** typical.

The goal is **fair comparison** between hypotheses of different lengths — not always preferring long.

---

# Beam search · keep top-$k$ paths

At each step, maintain $k$ candidate prefixes. Expand each by all next tokens, keep the top $k$ by **length-normalized** log-prob score.

Typical $k = 4$ to $10$. More = better quality, slower. Still deterministic given $k$.

<div class="keypoint">

When output is bad, ask which error you have · **search error** — a high-probability sequence exists but beam missed it (bigger $k$ helps) · **model error** — the model assigns its highest probability to garbage (no beam width can fix that).

</div>

---

# ⭐⭐⭐ Optional · beam search · worked example with $k=2$

Vocab · {The, A, cat, dog, sat, ran}. Decoding "The cat sat".

<div class="math-box">

**Step 1.** Beams · `[<s>]`. Top-2 next tokens · **A** (−0.5) · **The** (−0.7). Keep both.

**Step 2.** Expand each beam; top extensions:

| sequence | logP |
|:-:|:-:|
| `<s> The cat` | −0.7 + −0.6 = **−1.3** |
| `<s> The dog` | −0.7 + −1.3 = −2.0 |
| `<s> A dog` | −0.5 + −0.9 = −1.4 |
| `<s> A feline` | −0.5 + −1.1 = −1.6 |

Keep the top 2 · `<s> The cat` (−1.3) and `<s> A dog` (−1.4).

**Step 3.** Continue. Final score divides by $T^{0.6}$ to compare different lengths.

</div>

Greedy would have committed to "A" at step 1 (locally best) and ended at "A dog ran" (logP −2.5). The beam kept "The…" alive and finds "The cat sat" (logP −1.7).

---

# Top-$k$ and nucleus · the improv-comedian analogy

<div class="insight">

- **Greedy / beam** · a predictable comedian who always says the most obvious next line. Safe, but boring and repetitive.
- **Sampling** · a creative comedian who knows the top 5–10 good responses and sometimes picks the 3rd or 4th — leads to a funnier story.

Top-$k$ and nucleus are two ways to define the **pool of good options** for the comedian to sample from.

</div>

---

# Top-$k$ vs nucleus · how the pool is chosen

For open-ended generation (story writing, chat) beam is *too* deterministic — everything sounds the same.

- **Top-$k$** · pool = the $k$ most-likely tokens. Fixed size. Renormalize, sample.
- **Top-$p$ (nucleus)** · pool = smallest set with cumulative prob $\geq p$. **Dynamic size** — small when the model is confident, large when uncertain.

When the next-token distribution has one tall bar, nucleus narrows automatically; when it's flat, nucleus widens. Top-$k$ ignores this and uses the same $k$ everywhere.

<div class="realworld">

**Current LLM default** — nucleus with $p = 0.9$ or $0.95$, plus a temperature knob.

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

What survived into 2026?

---

# Where Seq2Seq-like ideas still ship

- **Speech recognition** · Whisper uses encoder-decoder with attention (the full recipe).
- **Machine translation** · Transformer encoder-decoder is *literally* Seq2Seq with attention instead of a context vector.
- **Code generation** · same family, different data.
- **Summarization** · T5, BART — encoder-decoder Transformers.

<div class="realworld">

The Seq2Seq **pattern** (encoder → context → decoder) is everywhere. Only the *implementation* of "context" changed: fixed vector (2014) → attention (2015) → self-attention (2017) → ... → your favorite LLM today.

</div>

---

# Putting it all together · the L11 master sentence

<div class="math-box">

**Seq2Seq = encoder squeezes input into a vector · decoder unspools that vector into the output.** It works for short sentences and breaks for long ones · the fixed-size bottleneck can't hold everything. That failure is what *demands* attention in L12.

</div>

| Piece | Role | Friendly intuition |
|:-:|:-:|:-:|
| Encoder RNN | reads source sequence | "memorize what was said" |
| Context $\mathbf{c}$ | final hidden state | "the gist" |
| Decoder RNN | generates target sequence | "say it in the new language" |
| Teacher forcing | feed true previous token | "read along while learning" |
| Beam search | top-$k$ partial hypotheses | "keep $k$ best drafts" |

This is **literally how Google Translate worked from 2014 to 2016** — and what made attention so urgent.

---

# Pop quiz · revisit

The 14-word relative-clause sentence into Hindi?

<div class="keypoint">

(a) Word-by-word fails · Hindi reorders the clauses — there is no per-word alignment to follow.
(b) The Seq2Seq of today's lecture · works, but the *whole* nested sentence must squeeze through one fixed vector — exactly the bottleneck that makes long sentences degrade.
(c) ✓ Looking back at any source word while writing each target word — that's **attention**, and it's the entire subject of L12.

</div>

You now know precisely *why* (b) breaks. Next lecture builds (c).

---

# Practice problems

<div class="math-box">

**P1.** Why is BLEU computed on $n$-grams ($n=1\ldots4$) rather than just on word matches? Sketch a translation that has high 1-gram precision but low BLEU.

**P2.** Teacher forcing trains $p(y_t \mid y_{<t}^*, x)$ but inference uses $p(y_t \mid \hat y_{<t}, x)$. Name this distribution mismatch and one common mitigation (scheduled sampling).

**P3.** Beam search with width $k = 5$ and vocabulary $V = 50{,}000$. How many candidate continuations are scored per step? Per generation step the top $k$ are kept — total cost for a 30-token output?

</div>

---

# Practice problems · bottleneck &amp; decoding

<div class="math-box">

**P4.** Show that for a fixed-size context vector $\mathbf{c} \in \mathbb{R}^{512}$, the encoder is *information-theoretically* limited · roughly how many bits of the source sentence can it preserve?

**P5.** Greedy decoding can produce hallucinated repetitions (`I I I I…`). Why? What does **temperature > 1** or **top-$p$ sampling** do to fix this in practice?

**P6.** A Seq2Seq model on a 50-word sentence has the encoder's *last* hidden state as context. Explain why this is the worst possible choice when the *answer* depends on word #1 (e.g. subject-verb agreement at the end).

</div>

---

# What breaks · symptom → suspect → test

| Symptom | Suspect | Fastest test |
|---|---|---|
| Great on short sentences, garbage past ~30 tokens | fixed-length context bottleneck | plot BLEU vs source length — look for the cliff |
| Low training loss, but generation derails after one wrong word | exposure bias — teacher forcing at train, autoregressive at test | decode *training* pairs autoregressively, watch the cascade |
| Beam search always returns suspiciously short outputs | no length normalization — log-probs punish length | rescore with $\tfrac{1}{T^{0.6}}\sum\log P$, compare winners |
| Decoder loops · "I I I I…" | greedy decoding stuck in a repetition attractor | switch to nucleus $p=0.9$ + temperature, regenerate |
| Output is `<unk>` or empty from token 1 | decoder input not shifted right / missing `<s>` | print the first decoder input batch — it must start with `<s>` |

---

<!-- _class: summary-slide -->

# Lecture 11 — summary

- **Seq2Seq** · encoder-decoder for variable-length input → variable-length output.
- **Context vector** is the encoder's final hidden state — a compressed summary.
- **Teacher forcing** trains the decoder with ground-truth history; **exposure bias** is the price.
- **Decoding** — greedy, beam (length-normalized), top-k, **nucleus (the current default)**.
- **The bottleneck** — one fixed vector for all source info; BLEU collapses past ~30 tokens.
- **Attention** is the fix · next lecture opens Module 6.

### Read before Lecture 12

**Prince Ch 12** (early sections on attention and QKV).

### Next lecture

**The Attention Mechanism** — Bahdanau additive · Luong multiplicative · the query-key-value abstraction · scaled dot-product and why $\sqrt{d_k}$ · self-attention.

<div class="notebook">

**Notebook 11** · `11-seq2seq-nmt.ipynb` — tiny English→French translator with an LSTM encoder-decoder; no attention yet. Next notebook adds attention and you'll see the BLEU gap close.

</div>

---

# The one-sentence takeaway

<div class="insight">

**If the whole sentence must squeeze through one vector, long sentences will always lose.**

</div>

*Next: stop compressing — let the decoder look back at every word. Attention.*
