---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Tokenization &amp; Pretraining Paradigms

## Lecture 14 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Where we are

Last lecture: the **Transformer block**. Stack it, add positional encoding, mask if autoregressive.

But stacking requires *inputs*. And inputs are **discrete symbols** (characters, subwords, words), not vectors.

<div class="paper">

Today maps to **Prince Ch 12 (pretraining)** + Karpathy's *Let's build the GPT Tokenizer* video. Tokenization is the part of LLMs everyone wants to skip — don't.

</div>

Four questions:

1. Why is tokenization hard?
2. How does **BPE** work step-by-step?
3. What are the three **pretraining paradigms** — BERT, GPT, T5?
4. Why does tokenization cause so many LLM bugs?

---

<!-- _class: section-divider -->

### PART 1

# Why tokenization is hard

Spoiler · there's no good answer

---

# Three failed alternatives

| Unit | Problem |
|------|---------|
| **Characters** | sequences are 5–10× longer → attention becomes expensive ($O(N^2)$) |
| **Whole words** | vocab must be ~$10^6$; unseen words → OOV; misspellings fail |
| **Morphemes** | language-specific; requires linguistic annotation; doesn't scale |

<div class="keypoint">

Subword tokenization is the compromise: common words as single tokens, rare words split into learned subwords.

</div>

---

# A concrete failure · character LMs

Train a character-level Transformer on Wikipedia. It *works* — GPT-like samples of readable English.

But it's expensive:
- A 1000-word page → ~6000 characters → 6000 attention positions → **36M attention entries per layer**.
- Equivalent word-level model: 1000 tokens → **1M entries**. 36× cheaper.

Plus · character LMs have to *learn* that "t-h-e" is a word, unit by unit. That's capacity wasted on a solved problem.

<div class="insight">

Subwords compromise: keep common sequences as one unit (saving sequence length), split rare sequences (keeping open vocabulary). The winning middle path.

</div>

---

# The sweet spot · subwords

Ideal subword tokenizer:
- Common words → single tokens (cheap, frequent)
- Rare words → composition of familiar subwords (generalizable)
- **No OOV** — any byte sequence is tokenizable
- Vocab size tunable (typically 30k–100k)

The winning algorithm: **Byte-Pair Encoding** (BPE), re-purposed from 1994 data compression.

---

<!-- _class: section-divider -->

### PART 2

# BPE step-by-step

One merge rule at a time

---

# BPE merges · visual

![w:920px](figures/lec14/svg/bpe_merges.svg)

<div class="realworld">

▶ Interactive: type a corpus, press "merge" to see the most frequent pair get glued live — [bpe-merges](https://nipunbatra.github.io/interactive-articles/bpe-merges/).

</div>

---

# The BPE algorithm · 7 lines

```python
def train_bpe(corpus, n_merges):
    # 1. Split every word into characters
    tokens = [list(word) for word in corpus.split()]
    merges = []

    for _ in range(n_merges):
        # 2. Count all adjacent pairs
        pair_counts = Counter((a, b) for word in tokens for a, b in zip(word[:-1], word[1:]))
        if not pair_counts: break

        # 3. Find the most frequent pair
        best = pair_counts.most_common(1)[0][0]
        merges.append(best)

        # 4. Apply the merge across all tokens
        tokens = [merge_in_word(w, best) for w in tokens]

    return merges, tokens
```

At inference, apply the same merge rules in order → tokenize any new string.

---

# Worked BPE · "low lower newest widest"

Start at the character level:

```
"l o w", "l o w e r", "n e w e s t", "w i d e s t"
```

Count pairs: `(e, s)` appears 2×, `(s, t)` appears 2×, `(l, o)` appears 2×, `(o, w)` appears 2×...

<div class="math-box">

**Merge 1**: `(e, s) → es` · tokens: `l o w`, `l o w e r`, `n e w es t`, `w i d es t`
**Merge 2**: `(es, t) → est` · tokens: `l o w`, `l o w e r`, `n e w est`, `w i d est`
**Merge 3**: `(l, o) → lo` · tokens: `lo w`, `lo w e r`, `n e w est`, `w i d est`
**Merge 4**: `(lo, w) → low` · tokens: `low`, `low e r`, `n e w est`, `w i d est`

</div>

With four merges we've built `low`, `est` as single tokens — exactly the reusable subwords. At inference, apply merges 1-4 in order to any new word.

---

# Why byte-level BPE is the default

Two breakthroughs GPT-2 introduced:

1. **Start from bytes (0–255), not Unicode characters.** Every possible string becomes tokenizable, including emojis, foreign scripts, binary garbage.
2. **Pretokenize by regex** before BPE, to avoid crossing word boundaries ("New York" stays as two separate merge chains).

<div class="keypoint">

Result · a 50k-token vocab that covers English, code, Japanese, emoji, and anything else users throw at it. No `<unk>` token needed.

</div>

Llama, GPT-*, Mistral, Claude all use byte-level BPE with minor tweaks. SentencePiece is the same idea packaged for cross-language training.

---

# Three BPE variants you will meet

| Variant | How | Used in |
|---------|-----|---------|
| **Character-level BPE** | start from Unicode chars | original 2015 paper |
| **Byte-level BPE** | start from raw bytes | GPT-2, Llama, most modern LLMs |
| **WordPiece** | same idea, likelihood-based merge | BERT, DistilBERT |
| **SentencePiece** | treat whitespace as regular char | Llama, mT5, multilingual |

<div class="insight">

Byte-level BPE (GPT-2) is now the default · handles any unicode, any language, any emoji, no OOV.

</div>

---

# Tokenization gotchas · real LLM failures

<div class="warning">

**"How many r's in strawberry?"** — GPT-4 famously miscounted. Why? "strawberry" tokenizes to something like ["straw", "berry"] or ["str", "aw", "berry"]. The model never sees individual letters — it sees chunks.

**Arithmetic errors.** Numbers tokenize inconsistently: "1234" might be one token, "1235" might split. Models learn arithmetic by memorizing token patterns, not digit manipulation.

**Spaces matter.** " the" (with leading space) is a different token from "the". This is why prompts to LLMs are sensitive to trailing spaces.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Three pretraining paradigms

Same Transformer · different objectives

---

# Three families · one architecture

![w:920px](figures/lec14/svg/mlm_vs_causal.svg)

---

# BERT · encoder-only · masked LM

Objective: predict masked tokens from **both left and right context**.

<div class="math-box">

**Masked Language Modeling (MLM)** · Devlin et al. 2018.

Randomly mask 15% of input tokens. Ask the model to predict them:

$$\mathcal{L}_\text{MLM} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\setminus i})$$

The model sees the WHOLE sentence (no causal mask) → rich bidirectional context.

</div>

Great for: classification, NER, retrieval (embeddings).
Bad for: generation — can't autoregressively extend.

---

# BERT · why mask 15%?

- Mask **too few** · most sequences see no loss signal → slow training.
- Mask **too many** · target is too hard, context too sparse.

15% was found empirically. Of those 15%:
- 80% actually replaced by `[MASK]`
- 10% replaced by a random token (adds noise, helps robustness)
- 10% left unchanged (so the model can't cheat by ignoring unmasked positions)

<div class="insight">

This mask-then-reconstruct recipe is the same idea as the denoising autoencoder from L19 — BERT is essentially a denoising autoencoder over language, using a Transformer encoder as the denoiser.

</div>

---

# GPT · decoder-only · causal LM

Objective: predict the **next token** given all previous tokens.

<div class="math-box">

**Causal Language Modeling (CLM)** · Radford et al. 2018.

$$\mathcal{L}_\text{CLM} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$$

Causal attention mask → model can only look backward.

</div>

Great for: generation, chat, code, anything where you produce text one token at a time.
Bad for: bidirectional understanding tasks (but at scale, GPT-3+ closed this gap anyway).

---

# GPT · why causal loss is so rich

One tiny objective — predict the next token — forces the model to reason about:

- **Syntax** · closing brackets, matching tenses, agreement
- **Semantics** · what words follow each other meaningfully
- **World knowledge** · "The capital of France is …" requires a fact
- **Reasoning** · "If all A are B, and x is A, then x is …" requires logic
- **Style** · code continuations, formal-register, poetry

<div class="keypoint">

Every position in a 2048-token window is a little training example. A 1T-token corpus gives you $10^{12}$ supervised tasks for free — no human labeling needed.

</div>

This scale-and-generality combo is why *next-token prediction* — despite looking trivial — ended up subsuming most of NLP.

---

# T5 · encoder-decoder · text-to-text

Frame every task as text-to-text:

```
"translate English to German: The house is wonderful."
   → "Das Haus ist wunderbar."

"summarize: ‹paragraph›"
   → ‹summary›

"question: ‹q›  context: ‹c›"
   → ‹answer›
```

<div class="paper">

Raffel et al. 2019 · T5 — *Text-to-Text Transfer Transformer*. Unified framework; same model does translation, summarization, QA, classification.

</div>

Survives in some translation pipelines. But for pure generation, decoder-only (GPT pattern) won.

---

# Scaling · which paradigm won?

| Year | Winner | Why |
|------|--------|-----|
| 2018 | BERT | cheap, works, encoder embeddings useful |
| 2020 | GPT-3 | scale unlocked few-shot learning |
| 2022 | GPT-3.5 / InstructGPT | alignment via instruction tuning + RLHF |
| 2023 | GPT-4, Llama 2 | decoder-only becomes the dominant paradigm |
| 2026 | decoder-only + tool use + reasoning | everyone converges here |

<div class="realworld">

**Decoder-only** won the LLM race. BERT still ships in retrieval pipelines (small, fast, good embeddings). T5 survives where structured I/O matters.

</div>

---

<!-- _class: section-divider -->

### PART 4

# What pretraining actually learns

The "foundation" in foundation model

---

# Why pretraining works so well

Predicting the next token on a trillion-token corpus forces the model to learn:

- **Grammar** — what syntactic constructions are valid
- **Facts** — what follows "The capital of France is"
- **Reasoning patterns** — what tokens usually complete "if A then"
- **Style** — code, poetry, legal writing, instructions

<div class="keypoint">

Next-token prediction is so rich a task that a model good at it ends up learning **most of what's in the data** — implicitly, without any labeled supervision.

</div>

This is the "foundation" in foundation model.

---

# Scaling recap · 5 years of LLMs in one chart

| Year | Model | Params | Tokens | Notable |
|:-:|:-:|:-:|:-:|:-:|
| 2018 | BERT-base | 110M | 3.3B | first pretrained Transformer in production |
| 2019 | GPT-2 | 1.5B | 40B | "too dangerous to release" |
| 2020 | GPT-3 | 175B | 300B | first few-shot emergence |
| 2022 | Chinchilla | 70B | 1.4T | train-compute optimal |
| 2023 | Llama 2 70B | 70B | 2T | open weights, Chinchilla-ish |
| 2024 | Llama 3 8B | 8B | 15T | aggressively over-trained for inference |
| 2026 | frontier LLMs | 1T+ | 10T+ | multi-modal, reasoning, tool use |

<div class="insight">

Architecture has barely changed (decoder-only Transformer). What scaled: compute, data, careful engineering. "Attention + scale" really was the thing. L15 goes into the mechanical details.

</div>

---

# Fine-tuning · from pretrained to useful

Pretrained models are knowledgeable but not steerable. To make them follow instructions, chat, or specialize on a task:

1. **Supervised fine-tuning (SFT)** on instruction-response pairs.
2. **LoRA / QLoRA** — parameter-efficient fine-tuning (next lecture).
3. **RLHF / DPO** — align with human preferences (L16).

The pretrained model is the brain. Fine-tuning is how you train it to do what you want.

---

<!-- _class: summary-slide -->

# Lecture 14 — summary

- **Tokenization** balances sequence length vs vocab size. Subword wins.
- **BPE** · greedy adjacent-pair merges; byte-level variant (GPT-2, Llama) has no OOV.
- **Tokenization bugs** · spelling errors, arithmetic weirdness, space sensitivity — all trace to token boundaries.
- **BERT** · encoder-only · bidirectional MLM · classification + embeddings.
- **GPT** · decoder-only · causal LM · generation. **The winner for scaled LLMs.**
- **T5** · encoder-decoder · text-to-text framing. Niche role in 2026.

### Read before Lecture 15

Chinchilla paper (Hoffmann 2022) + HuggingFace course Chapter 1.

### Next lecture

**Large Language Models** — Chinchilla scaling, RoPE, GQA, distributed training, emergent abilities.

<div class="notebook">

**Notebook 14** · `14-bpe-from-scratch.ipynb` — implement BPE tokenizer from scratch; train on a small corpus; visualize merges; tokenize new sentences.

</div>
