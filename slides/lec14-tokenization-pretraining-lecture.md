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

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain why **tokenization** is the "first design decision of an LLM."
2. Run the **BPE** algorithm on paper for 5 merges.
3. Distinguish **character / word / subword / byte-BPE** and pick appropriately.
4. Identify common **tokenization bugs** (counting letters, arithmetic, whitespace).
5. Contrast the three **pretraining paradigms** · BERT · GPT · T5.
6. Name the **data + compute economics** of a 70B-param training run.

---

# Where we are

Last lecture (L13): the **Transformer block**. Stack it, add positional encoding, mask if autoregressive.

But stacking requires *inputs*. And inputs are **discrete symbols** (characters, subwords, words), not vectors.

<div class="paper">

**Reading & inspiration** ·
- **Karpathy · *Let's build the GPT Tokenizer*** (YouTube) — best 2-hour walk-through.
- **Hugging Face course · Ch 6 (Tokenizers)** — code-first, free.
- **Hugging Face course · Ch 1–2** — pretraining paradigms (BERT, GPT, T5).
- **UDL (Prince) · Ch 12 (pretraining)**.

Tokenization is the part of LLMs everyone wants to skip — don't.

</div>

---

# Bridge from ES 335 · you already know the objective

In ES 335 you built a **next-token-prediction** model · given the words so far, predict the next one by minimizing cross-entropy — the same cost $J(\theta)=-\sum_t \log \hat y_{t}$ where $\hat y_t = P_\theta(x_t \mid x_{<t})$.

<div class="insight">

The *objective* hasn't changed since ES 335 — it's still next-token cross-entropy $J(\theta)$. What changes today is everything *around* it · (1) **how raw text becomes the discrete tokens** $x_t$ in the first place (tokenization / BPE), and (2) **what happens when you scale** that one objective to a trillion tokens (pretraining: BERT vs GPT vs T5).

</div>

---

# Four questions

1. Why is tokenization hard?
2. How does **BPE** work step-by-step?
3. What are the three **pretraining paradigms** — BERT, GPT, T5?
4. Why does tokenization cause so many LLM bugs?

---

# Pop quiz · how many "tokens" is this sentence?

Take · **"GPT-4 is great!"**

<div class="popquiz">

(a) 4 tokens (one per word).
(b) 5 tokens (split punctuation).
(c) 7 tokens · `"G", "PT", "-", "4", " is", " great", "!"`.
(d) Different number depending on the model.

Stop and guess — and hold that guess. By the end of Part 2 you'll be able to *derive* the answer, and it explains why "how many r's in strawberry?" was a well-known failure mode for years.

</div>

---

▶ **Interactive** · watch BPE merges grow on real text → [bpe-merges](https://nipunbatra.github.io/interactive-articles/bpe-merges/).

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

---

# Smart-dictionary analogy

<div class="keypoint">

A tokenizer builds a **dictionary** for the language.

- Character dictionary · just the alphabet · too basic.
- Word dictionary · huge · breaks on "un-un-believable".
- **Subword dictionary** · common words plus reusable prefixes (`un-`) and suffixes (`-able`) · can compose any new word.

</div>

That's the sweet spot we'll build with BPE on the next slides.

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

# Worked BPE · merge trace

![w:920px](figures/lec14/svg/bpe_merge_trace.svg)

---

# BPE · the data-compression analogy

<div class="insight">

Like a basic file compressor. If "ABCABCABC" appears a lot, define a new symbol $Z = $ "ABC" and rewrite as "ZZZ". BPE does the same for language · find the most common adjacent pair (like `th`), compress it into a single new token, repeat.

</div>

---

# Worked BPE · "low lower newest widest"

**Step 0 · split each word into characters with `</w>` end marker.**
`l o w </w>`, `l o w e r </w>`, `n e w e s t </w>`, `w i d e s t </w>`

**Step 1 · count adjacent pairs.**
`(l, o): 2`, `(o, w): 2`, `(w, e): 2`, `(e, s): 2`, `(s, t): 2`, `(t, </w>): 2`, `(e, r): 1`, …
Six-way tie. Break it however you like — say, pick `(e, s)`.

**Merge 1 · `(e, s) → "es"`.**
`l o w </w>`, `l o w e r </w>`, `n e w es t </w>`, `w i d es t </w>`

**Step 2 · recount.** `(es, t): 2` is a *new* pair, tied with `(l,o), (o,w), (t,</w>)`. Pick `(es, t)`.

**Merge 2 · `(es, t) → "est"`.**
`l o w </w>`, `l o w e r </w>`, `n e w est </w>`, `w i d est </w>`

---

# Worked BPE · merges 3–4 and the result

**Merge 3 · `(l, o) → "lo"`** (count 2) · then **Merge 4 · `(lo, w) → "low"`** (count 2).

`low </w>`, `low e r </w>`, `n e w est </w>`, `w i d est </w>`

After 4 merges the vocabulary has learned `low` and `est` as single tokens — a whole common word and a reusable suffix, discovered purely from counts.

At inference, apply merges 1–4 **in the same order** to any new word: `lowest` → `low est </w>` — a word never seen in training, tokenized into two familiar units.

---

# Worked BPE · second example

Corpus · `"hug bug rug"`. Initial: `h u g </w>`, `b u g </w>`, `r u g </w>`.

**Count pairs.** $(u, g): 3$, $(g, \langle/w\rangle): 3$, $(h, u), (b, u), (r, u)$ each $1$.

**Merge 1 · $(u, g) \to$ "ug".** New: `h ug </w>`, `b ug </w>`, `r ug </w>`.

**Recount.** $(ug, \langle/w\rangle): 3$, others 1.

**Merge 2 · $(ug, \langle/w\rangle) \to$ "ug</w>".** Final: `h ug</w>`, `b ug</w>`, `r ug</w>`.

The algorithm has discovered the reusable suffix `ug` and the morphologically meaningful unit `ug</w>`.

---

# Why byte-level BPE is the default

Two breakthroughs GPT-2 introduced:

1. **Start from bytes (0–255), not Unicode characters.** Every possible string becomes tokenizable, including emojis, foreign scripts, binary garbage.
2. **Pretokenize by regex** before BPE, to avoid crossing word boundaries ("New York" stays as two separate merge chains).

<div class="keypoint">

Result · a 50k-token vocab that covers English, code, Japanese, emoji, and anything else users throw at it. No `<unk>` token needed.

</div>

Most modern LLMs — Llama, GPT-*, Mistral — use byte-level BPE with minor tweaks. SentencePiece is the same idea packaged for cross-language training.

**Trade-off** · a vocab trained mostly on English splits other languages into far more tokens (higher *fertility*) — non-English users pay more context and more cost per word.

---

# Tokenizer comparison · same sentence, different counts

![w:920px](figures/lec14/svg/tokenizer_comparison.svg)

---

# Four tokenizers you will meet

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

# BERT vs GPT · side-by-side

![w:920px](figures/lec14/svg/mlm_vs_causal_detail.svg)

---

# BERT · the cloze-test analogy

<div class="keypoint">

BERT plays a **fill-in-the-blanks** game · like a cloze test in school.

We hide ~15% of the words; BERT must guess them. Because it can see text **on both sides** of the blank, it gets very good at *understanding* context.

</div>

This makes BERT a strong **encoder** · ideal for tasks where you need a representation of the whole sentence (classification, retrieval, NER). It's bad at *generating* text · because it never practices producing tokens one-by-one.

---

# BERT · the cloze-test math, step by step

1. **Sentence.** $x = [\text{The},\text{cat},\text{sat},\text{on},\text{the},\text{mat}]$.
2. **Mask token 4.** $x' = [\text{The},\text{cat},\text{sat},[\text{MASK}],\text{the},\text{mat}]$.
3. **Goal.** Predict the original token from context. We want to maximize $P(x_4 = \text{"on"} \mid x_{\setminus 4})$.
4. **Loss.** Standard trick · negative log probability:
$$\text{loss}_4 = -\log P(x_4 \mid x_{\setminus 4})$$
5. **Total.** Sum over the ~15% masked positions:
$$\mathcal{L}_\text{MLM} = \sum_{i \in \text{masked}} -\log P(x_i \mid x_{\setminus i})$$

The model sees the whole sentence (no causal mask) → rich bidirectional context.

---

# Worked numeric · BERT loss for one mask

Input · `…sat [MASK] the…`. Correct answer · "on".

Transformer outputs logits over vocab at the `[MASK]` position:
- $\text{logit}(\text{on}) = 3.5$, $\text{logit}(\text{above}) = 2.1$, $\text{logit}(\text{under}) = 1.5$, …

Softmax · $\exp(3.5) \approx 33.1$, $\exp(2.1) \approx 8.2$, $\exp(1.5) \approx 4.5$ (rest of vocab ≈ negligible).
$P(\text{on}) \approx 33.1 / (33.1 + 8.2 + 4.5) \approx \mathbf{0.72}$.

Loss · $-\log 0.72 \approx \mathbf{0.33}$.

If the model were more confident ($P = 0.99$), loss would be $-\log 0.99 \approx 0.01$ — model rewarded for confidence on the right answer.

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

# GPT · the smartphone-keyboard analogy

<div class="insight">

Predictive text on your phone. Type *"I am heading to the…"* — it suggests "gym", "store", "movies". Predicts the **next word** from what you've already typed; never sees the future.

GPT does this for every word, learning to be an excellent generator.

</div>

---

# GPT · CLM math, step by step

Sentence · $x = [\text{The}, \text{cat}, \text{sat}]$. Break into prediction problems:
1. Given `<s>`, predict "The". Loss · $-\log P(\text{The})$.
2. Given "The", predict "cat". Loss · $-\log P(\text{cat} \mid \text{The})$.
3. Given "The cat", predict "sat". Loss · $-\log P(\text{sat} \mid \text{The}, \text{cat})$.

Total · sum over all positions:
$$\mathcal{L}_\text{CLM} = \sum_{t=1}^{T} -\log P(x_t \mid x_{<t})$$

Causal attention mask · model can only look backward.

---

# Worked numeric · GPT loss at one step

Context · "The cat …". Correct next word · "sat".

Logits over vocab · $\text{logit}(\text{sat}) = 4.0$, $\text{logit}(\text{ran}) = 2.5$, $\text{logit}(\text{jumped}) = 2.0$, …

Softmax · $\exp(4.0) \approx 54.6$, $\exp(2.5) \approx 12.2$, $\exp(2.0) \approx 7.4$ (rest of vocab ≈ negligible).
$P(\text{sat} \mid \text{The cat}) \approx 54.6 / (54.6 + 12.2 + 7.4) \approx \mathbf{0.74}$.

Loss at $t = 3$ · $-\log 0.74 \approx \mathbf{0.30}$.

Total sentence loss · sum these up across every position. A 2048-token window gives 2048 little training problems for free, every step.

Great for: generation, chat, code, anything where you produce text one token at a time.
Bad for: bidirectional understanding (but at scale, GPT-3+ closed this gap).

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

Its pretraining objective is **span corruption** · mask out contiguous *spans* of tokens, train the decoder to reconstruct them — MLM's idea, adapted to encoder-decoder.

Survives in some translation pipelines. But for pure generation, decoder-only (GPT pattern) won.

---

# Scaling · which paradigm won?

| Year | Winner | Why |
|------|--------|-----|
| 2018 | BERT | cheap, works, encoder embeddings useful |
| 2020 | GPT-3 | scale unlocked few-shot learning |
| 2022 | GPT-3.5 / InstructGPT | alignment via instruction tuning + RLHF |
| 2023 | GPT-4, Llama 2 | decoder-only becomes the dominant paradigm |
| 2026 | decoder-only + tool use + reasoning | the current trajectory |

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

---

# Pretraining data · where does 15T tokens come from?

<div class="columns">
<div>

### Web-scale sources

- **Common Crawl** · ~2T web pages (filtered)
- **C4** · curated Common Crawl · 750B tokens
- **RefinedWeb** · quality-filtered · 5T tokens
- **Wikipedia** · 10B tokens (20 languages)

</div>
<div>

### Specialized

- **GitHub code** · 1-2T tokens (de-licensed)
- **ArXiv + books3 + StackExchange** · 100B+
- **Synthetic data** (Phi-3) · textbooks generated by stronger models
- **Multilingual scraped** · CCMatrix, mC4

</div>
</div>

The headline number (15T) is dominated by filtered web text; everything else is seasoning.

---

# Pretraining data · the legal question

<div class="warning">

**Copyright issues** · training on copyrighted text without a license is legally contested. *NY Times v. OpenAI* (filed 2023) · still unresolved. The data pipeline is as much a legal project as a technical one.

</div>

Practical fallout · several labs now disclose data sources, license corpora, or lean harder on **synthetic** and **licensed** data to reduce exposure.

---

# Data quality beats quantity

Phi-3 (2024) · trained on **3T** "textbook-quality" tokens (heavily filtered + synthetic); matched Llama-2 7B trained on **2T** web tokens.

<div class="math-box">

| Recipe | Tokens | Quality | Outcome |
|:-:|:-:|:-:|:-:|
| 15T web (Llama-3) | large, noisy | baseline | strong LLM |
| 3T curated (Phi-3) | small, high | synth | matches a larger model |
| 1T books (Books3) | medium, high | literary | great for story writing |

</div>

<div class="keypoint">

**Data curation is now the hottest research area.** Dedup, language-ID, toxicity filter, repetition filter, perplexity filter with smaller model, synthetic augmentation. Each step buys 1-5 points on benchmarks.

</div>

---

# ⭐⭐⭐ Optional · Compute economics · where the 6 comes from

The famous LLM rule of thumb: **training FLOPs $\approx 6 \cdot N \cdot D$** where $N$ = parameters, $D$ = training tokens.

**Why 6?** Per token:
1. **Forward pass.** Most compute is the FFN's two big matmuls. ≈ $\mathbf{2N}$ FLOPs per token.
2. **Backward pass.** Standard rule · ~2× the forward cost. ≈ $\mathbf{4N}$ FLOPs per token.
3. **Total per token** · $2N + 4N = \mathbf{6N}$.

Multiply by all $D$ tokens in the dataset:
$$\text{FLOPs} \approx 6 \cdot N \cdot D$$

---

# Worked numeric · 70B model training cost

One training run of a 70B-parameter model on Chinchilla-optimal $D = 1.4$T tokens:
- FLOPs $\approx 6 \cdot (7 \cdot 10^{10}) \cdot (1.4 \cdot 10^{12}) \approx 5.9 \times 10^{23}$
- Time · $5.9 \times 10^{23} / (4000 \cdot 1.5 \times 10^{14}) \approx 10^6$ s → **~11 days** on 4k A100s (at ~150 TFLOPS / GPU effective)
- Cost · ~1.1M GPU-hours → **~$5M** at commercial GPU-hour rates

**Smaller check · Llama 2 7B.** $N = 7 \times 10^9$, $D = 2 \times 10^{12}$:
$\text{FLOPs} \approx 6 \cdot 7 \cdot 10^9 \cdot 2 \cdot 10^{12} = 8.4 \times 10^{22}$.

Llama 3 70B · reportedly ~$80M including experiments. GPT-4 class · ~$100M+ per training run.

<div class="realworld">

10 years ago a deep net cost tens of dollars to train. Today, frontier model training costs a house.

</div>

---

# Scaling recap · 5 years of LLMs in one chart

| Year | Model | Params | Tokens | Notable |
|:-:|:-:|:-:|:-:|:-:|
| 2018 | BERT-base | 110M | 3.3B | first pretrained Transformer in production |
| 2019 | GPT-2 | 1.5B | ~10B | "too dangerous to release" |
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

# Putting it all together · the L14 master sentence

<div class="math-box">

**Tokenization is the first design decision of an LLM** · BPE merges the most common adjacent pairs until you have a 32k–128k vocabulary that balances "few short tokens" (efficient) and "no out-of-vocabulary" (robust). **Pretraining** then optimizes one of three NLL objectives over those tokens · **MLM** (BERT) for understanding, **CLM** (GPT) for generation, **span-corruption** (T5) for both.

</div>

| Paradigm | Loss | Best for |
|:-:|:-:|:-:|
| BERT (MLM) | predict masked tokens | classification, embeddings |
| GPT (CLM) | predict next token | generation, dialogue |
| T5 (span) | predict masked span | translation, summarization |

All three are **NLL of a Categorical** (L00) — the only difference is *which* tokens are masked.

---

# Pop quiz · revisit

"GPT-4 is great!" — how many tokens? Answer **(d) · it depends on the model's tokenizer**.

<div class="keypoint">

(a), (b) Word-level splits — no modern LLM tokenizes this way.
(c) Correct for *one particular* vocabulary — a GPT-2-style byte-level BPE gives exactly `"G", "PT", "-", "4", " is", " great", "!"` — but a different vocab merges differently.
(d) ✓ Token count is a property of the **tokenizer**, not the sentence. Same text, different model → different count, different context cost.

</div>

And the model never sees the letters *inside* those chunks — which is exactly why "how many r's in strawberry?" was hard.

---

# Practice problems

<div class="math-box">

**P1.** A corpus has 10k unique words but 1k unique subwords cover 95% of tokens. Argue why **subword** tokenization is the right compromise between *word* and *character*.

**P2.** Run BPE by hand on the corpus `"low low low lower newest"` for 3 merges. Show the vocabulary after each step.

**P3.** Why does GPT use a **causal mask** during training but BERT doesn't? Sketch the attention mask matrix for each.

</div>

---

# Practice problems · masking &amp; vocab design

<div class="math-box">

**P4.** A masked LM masks 15% of tokens, replaces 80% of those with `[MASK]`, 10% with random tokens, 10% leaves them unchanged. Why this 80/10/10 split?

**P5.** Token "**ing**" appears in `running`, `singing`, `meeting`. Why is BPE *certain* to learn `ing` as one token? Estimate after how many merges.

**P6.** A tokenizer assigns a separate token to every digit 0–9. Why is this a poor design for arithmetic? What does Llama-3's tokenizer do differently?

</div>

---

# What breaks · symptom → suspect → test

| Symptom | Suspect | Fastest test |
|---|---|---|
| Model can't count letters in a word | tokenization, not intelligence | print the token IDs for "strawberry" |
| Arithmetic fine on small numbers, breaks on big ones | numbers tokenize inconsistently | tokenize "1234" vs "12345", compare the splits |
| Same prompt, trailing space → different answer | " the" and "the" are different tokens | tokenize both variants, diff the IDs |
| Non-English input costs 3× tokens, context "shrinks" | tokenizer fertility — vocab trained on English | compute tokens-per-word for each language |
| BERT fine-tune aces classification, can't write a sentence | MLM objective — never trained to generate left-to-right | sample autoregressively from it, watch the mush |

---

<!-- _class: summary-slide -->

# Lecture 14 — summary

- **Tokenization** balances sequence length vs vocab size. Subword wins.
- **BPE** · greedy adjacent-pair merges; byte-level variant (GPT-2, Llama) has no OOV.
- **Tokenization bugs** · spelling errors, arithmetic weirdness, space sensitivity — all trace to token boundaries.
- **BERT** · encoder-only · bidirectional MLM · classification + embeddings.
- **GPT** · decoder-only · causal LM · generation. **The winner for scaled LLMs.**
- **T5** · encoder-decoder · text-to-text framing. Niche role today.

### Read before Lecture 15

Chinchilla paper (Hoffmann 2022) + HuggingFace course Chapter 1.

### Next lecture

**Large Language Models** — Chinchilla scaling, RoPE, GQA, distributed training, emergent abilities.

<div class="notebook">

**Notebook 14** · `14-bpe-from-scratch.ipynb` — implement BPE tokenizer from scratch; train on a small corpus; visualize merges; tokenize new sentences.

</div>

---

# The one-sentence takeaway

<div class="insight">

**LLMs see tokens, not letters — half their bugs start there.**

</div>

*Next: scale it — Chinchilla, RoPE, GQA, and what a trillion tokens buys you.*
