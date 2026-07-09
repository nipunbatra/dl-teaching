---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Tokenization & Pretraining

## Lecture 16 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A Transformer never reads *text*. Its first layer is an **embedding table** — one row per vocabulary entry, looked up by an **integer ID**. So before any math, every string must become a sequence of integers from a **fixed, finite** vocabulary. But language is open-ended. That tension is the whole lecture.

<div class="keypoint">

**Two moving parts, one loss.** **(1) Tokenization** turns any string into integer IDs by learning subwords with **BPE** — merge the most frequent adjacent pair, repeat. **(2) Pretraining** then optimizes a single objective — **cross-entropy over tokens** — where GPT, BERT, and T5 differ *only* in which tokens are hidden.

</div>

You built a next-token model in ES 335 and met cross-entropy in L1. Today: **where do the tokens $x_t$ come from, and what does predicting them actually teach a model?**

$$\text{text} \xrightarrow{\;\text{BPE}\;} \underbrace{\text{integer IDs}}_{\text{tokenization}} \xrightarrow{\;\text{Transformer}\;} \underbrace{-\sum_t \log P_\theta(x_t \mid \text{context})}_{\text{pretraining loss}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Today builds on:**
- **L1** — cross-entropy is the NLL of a Categorical; that *is* the LM loss
- **ES 335** — next-token prediction, $J(\theta)=-\sum_t \log \hat y_t$
- **L13–L15** — the Transformer block that consumes these tokens

</div>
<div>

**And feeds forward into:**
- **L17** — the *economics*: Chinchilla, the $6ND$ rule, what a trillion tokens costs (we keep only a one-line pointer today)
- **L18+** — instruction tuning, RLHF, alignment — all start from a pretrained base

</div>
</div>

<div class="insight">

One objective — predict the hidden token — is rich enough to make **grammar, facts, reasoning, and style** fall out for free. Half of "LLM magic," and half of "LLM bugs," trace back to these two slides.

</div>

---

<!-- _class: section-divider -->

## Part 1 · Text in, integers out

---

# The problem, precisely

The model's first operation is a lookup: token ID $\to$ embedding row. So the pipeline is fixed — **text $\to$ tokens $\to$ integer IDs**, and only the IDs ever reach the network.

![w:1000px](figures/lec14/svg/tokenizer_pipeline.svg)

<div class="keypoint">

The tokenizer's entire job: map *any* string to IDs from a **fixed** vocabulary, and **never choke** on something it has never seen — a new word, an emoji, a URL, a foreign script, raw code.

</div>

---

# Why no word list can ever be enough

Language is **open-ended**: new slang, typos, product names, hashtags, dozens of writing systems, source code. A fixed dictionary of *words* is guaranteed to meet a string it cannot represent — an **out-of-vocabulary** (OOV) failure.

<div class="insight">

The fix must let us **compose** unseen words from familiar pieces — the way `un-` + `believ` + `-able` builds a word no list needed to store. That is the subword idea, and the rest of Part 1 rules out the two obvious extremes first.

</div>

---

# Two extremes, both fail

| Unit | Vocabulary | Failure |
|------|-----------|---------|
| **Characters** | ~100 symbols | sequences 5–10× longer → attention is $O(N^2)$ → expensive; must *relearn* that `t-h-e` is a word |
| **Whole words** | ~$10^6$ | huge embedding table; every unseen word / typo is **OOV**; can't compose |

<div class="keypoint">

**Subwords are the compromise:** common words stay one token (cheap), rare words split into learned pieces (composable), and *nothing* is ever OOV.

</div>

---

# A concrete cost: character-level LMs

A character Transformer *works* — it produces readable English. But it is wasteful:

<div class="columns">
<div>

**A 1000-word page**
- ≈ 6000 characters → 6000 attention positions
- $6000^2 \approx \mathbf{36{,}000{,}000}$ attention entries / layer

</div>
<div>

**Same page, word-level**
- ≈ 1000 tokens → 1000 positions
- $1000^2 = \mathbf{1{,}000{,}000}$ entries / layer
- **≈ 36× cheaper**

</div>
</div>

On top of the cost, the character model burns capacity *learning* that `t-h-e` is one unit — a problem a good tokenizer solves for free.

---

# The sweet spot: a subword dictionary

<div class="keypoint">

Think of the tokenizer as a **dictionary** for the language:

- **Character dictionary** — just the alphabet. Too basic; everything is long.
- **Word dictionary** — enormous, and breaks on `un-un-believable`.
- **Subword dictionary** — common words *plus* reusable prefixes (`un-`) and suffixes (`-able`). Composes any new word.

</div>

Vocab size is a **tunable knob** (typically 32k–128k). The algorithm that builds it: **Byte-Pair Encoding (BPE)**, borrowed from 1994 data compression.

---

# Intuition · vocabulary size is a dial you tune

The Ng-style move: don't agonize philosophically over char-vs-word — treat **vocabulary size as a hyperparameter** and read the tradeoff off a table. Bigger vocab → **shorter** sequences (cheaper attention), but a **fatter** embedding table and more rare, under-trained tokens.

| Vocabulary | Size | Tokens / English word | Failure mode |
|:--|:--:|:--:|:--|
| **Characters** | ~256 | ~4–5 | sequences 5× longer |
| **Subword (small)** | 32k | ~1.3 | — *the sweet spot* |
| **Subword (large)** | 100k | ~1.1 | rare tokens under-trained |
| **Whole words** | ~1M | 1.0 | OOV, giant table |

<div class="insight">

Every modern LLM lives in the **32k–128k** band — large enough that common words collapse to one token, small enough that the embedding table and rare-token counts stay sane. "Just right" is a **measured** compromise, not a philosophy.

</div>

---

# Warm-up: how many tokens is this sentence?

Take **"GPT-4 is great!"**

<div class="popquiz">

(a) 4 — one per word · (b) 5 — split the punctuation · (c) 7 — `"G","PT","-","4"," is"," great","!"` · (d) it depends on the model.

</div>

*Answer:* **(d).** Token count is a property of the **tokenizer**, not the sentence. A GPT-2-style byte-level BPE happens to give exactly (c); a different vocabulary merges differently. Same text, different model → different count → different context cost. And notice: the model never sees the letters *inside* `"PT"` — hold that thought for "strawberry."

---

<!-- _class: section-divider -->

## Part 2 · Subwords via BPE

---

# BPE in one idea

<div class="keypoint">

**Start** from the raw characters. **Repeat:** find the **most frequent adjacent pair** of tokens and **merge** it into one new token. Frequent stuff (`th`, `ing`, whole common words) collapses to a single token; rare stuff stays split into a few subwords — so nothing is ever OOV.

</div>

![w:900px](figures/lec14/svg/bpe_merges.svg)

---

# The same trick a file compressor uses

<div class="insight">

If `ABCABCABC` appears often, define a new symbol $Z = $ `ABC` and rewrite it as `ZZZ`. BPE does exactly this for language: find the most common adjacent pair (like `t`+`h`), replace every occurrence with one new token `th`, and repeat. Compression and tokenization are the *same algorithm* — one shrinks files, the other shrinks sequences.

</div>

---

# Intuition · why merge the *most frequent* pair

Why greedily merge the **most frequent** pair, rather than the "most meaningful" one? Because language is deeply **Zipfian** — a tiny handful of pairs (`t`+`h`, `i`+`n`, `e`+`</w>`) are wildly more common than all the rest.

<div class="insight">

Merging the top pair first buys the **most compression per new token**: one merge of `t`+`h` shortens millions of positions at once. BPE spends its early, precious vocabulary slots exactly where the data is densest — **greedy compression guided by counts, with no hand-written rules.** A linguist would argue about morphemes; BPE just *counts*.

</div>

<div class="keypoint">

Same reason Huffman coding gives short codes to frequent symbols (L1): **frequency is where the bits are.** BPE is Huffman's cousin, run on adjacent *pairs* instead of single symbols.

</div>

---

# The BPE algorithm · 7 lines

```python
def train_bpe(corpus, n_merges):
    tokens = [list(word) + ["</w>"] for word in corpus.split()]   # 1. chars + end marker
    merges = []
    for _ in range(n_merges):
        pairs = Counter((a, b) for w in tokens                    # 2. count adjacent pairs
                        for a, b in zip(w[:-1], w[1:]))
        if not pairs: break
        best = pairs.most_common(1)[0][0]                          # 3. most frequent pair
        merges.append(best)
        tokens = [merge_in_word(w, best) for w in tokens]          # 4. apply the merge everywhere
    return merges, tokens
```

At **inference**, apply the learned merges **in the same order** to any new string. Training discovers the rules; encoding just replays them.

---

# Worked hand-trace · `low lower newest widest`

**Step 0 — split into characters + `</w>` end marker:**
`l o w </w>` · `l o w e r </w>` · `n e w e s t </w>` · `w i d e s t </w>`

**Step 1 — count adjacent pairs:** `(e,s):2`, `(s,t):2`, `(t,</w>):2`, `(l,o):2`, `(o,w):2`, `(e,r):1`, …
A tie at count 2 — break it any fixed way. Say we take **`(e,s)`**.

<div class="math-box">

**Merge 1 · `(e,s) → es`:** `l o w </w>` · `l o w e r </w>` · `n e w `**`es`**` t </w>` · `w i d `**`es`**` t </w>`

**Step 2 — recount:** `(es,t):2` is a *new* pair. Take it. **Merge 2 · `(es,t) → est`:** …`n e w `**`est`**` </w>` · `w i d `**`est`**` </w>`

</div>

---

# Worked hand-trace · finishing, and encoding a new word

**Merge 3 · `(l,o) → lo`** (count 2) → `lo w </w>` · `lo w e r </w>` · …
**Merge 4 · `(lo,w) → low`** (count 2) → **`low`**` </w>` · **`low`**` e r </w>` · `n e w est </w>` · `w i d est </w>`

<div class="insight">

Four merges, and BPE has discovered a **whole common word** (`low`) and a **reusable suffix** (`est`) — from counts alone, no linguist.

</div>

**Encoding a word never seen in training:** apply merges 1–4 in order to `lowest`:
$$\texttt{l o w e s t </w>} \xrightarrow{1,2} \texttt{l o w est </w>} \xrightarrow{3,4} \texttt{low est </w>}$$
A brand-new word, tokenized into two familiar units. **No OOV.**

---

# Worked numeric · what 4 merges bought

Count the tokens in the toy corpus `low lower newest widest`, **before** any merges and **after** the 4 merges (`es`, `est`, `lo`, `low`):

<div class="columns">
<div>

**Before — chars + `</w>`**
| word | tokens |
|:--|:--:|
| `low` | 4 |
| `lower` | 6 |
| `newest` | 7 |
| `widest` | 7 |
| **total** | **24** |

</div>
<div>

**After 4 merges**
| word | tokens |
|:--|:--:|
| `low·</w>` | 2 |
| `low·e·r·</w>` | 4 |
| `n·e·w·est·</w>` | 5 |
| `w·i·d·est·</w>` | 5 |
| **total** | **16** |

</div>
</div>

$$\text{compression} = \frac{24}{16} = \mathbf{1.5\times}\qquad(\text{sequence } 33\%\text{ shorter, for only } 4\text{ new tokens})$$

<div class="insight">

Real BPE runs **tens of thousands** of merges — the sequence keeps shrinking with sharply diminishing returns. That plateau is exactly why vocab size settles around 32k–128k (the dial from Part 1).

</div>

---

# The merge trace, visualized

![w:920px](figures/lec14/svg/bpe_merge_trace.svg)

Each row is one merge; the vocabulary grows by exactly one token per step, always gluing the current most-frequent pair.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** Corpus = `"ab ab abc"` (three words). Split each word into characters with a `</w>` end marker, then run **2 BPE merges** by hand. Show the pair counts and the tokenization after each merge. What single word is now one token?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**Step 0:** `a b </w>` (×2) · `a b c </w>` (×1).

**Merge 1 — count pairs:** `(a,b):3`, `(b,</w>):2`, `(b,c):1`, `(c,</w>):1`. Winner **`(a,b) → ab`**:
$$\texttt{ab </w>}\ (\times2)\quad\cdot\quad \texttt{ab c </w>}\ (\times1)$$

**Merge 2 — recount:** `(ab,</w>):2`, `(ab,c):1`, `(c,</w>):1`. Winner **`(ab,</w>) → ab</w>`**:
$$\texttt{ab</w>}\ (\times2)\quad\cdot\quad \texttt{ab c </w>}\ (\times1)$$

<div class="keypoint">

The word **`ab`** is now a single token. Note the `</w>` matters: `ab` at the *end* of a word became its own token, distinct from the `ab` inside `abc` — BPE learns word-boundary behavior for free.

</div>

---

# Byte-level BPE · the production default

<div class="keypoint">

Modern LLMs (GPT-\*, Llama, Mistral) run BPE over raw **bytes (0–255)**, not characters. Every possible string — any emoji, any script, code, binary junk — is a sequence of bytes, so **there is no `<unk>` token, ever.** A regex pre-split first stops merges from crossing word boundaries; SentencePiece packages the same idea for multilingual training.

</div>

<div class="warning">

**Consequence — fertility.** A vocabulary trained mostly on English splits other languages into *more* tokens. Non-English users pay more context and more money per word of the same meaning.

</div>

---

# Same sentence, different tokenizers

![w:920px](figures/lec14/svg/tokenizer_comparison.svg)

| Variant | How it starts | Used in |
|---------|--------------|---------|
| **Char-level BPE** | Unicode characters | original 2015/2016 paper |
| **Byte-level BPE** | raw bytes | GPT-2, Llama, most modern LLMs |
| **WordPiece** | likelihood-based merge | BERT |
| **SentencePiece** | whitespace is a normal char | Llama, mT5, multilingual |

---

# Tokenization gotchas · real LLM failures

<div class="warning">

**"How many r's in strawberry?"** — the model sees chunks like `["str","aw","berry"]`, never the individual letters. Counting characters means counting *inside* a token it treats as atomic.

**"Is 9.11 > 9.9?"** — `9.11` and `9.9` split into different token pieces (`9`,`.`,`11` vs `9`,`.`,`9`). The model compares learned token patterns, not decimal magnitudes.

**Spaces matter.** `" the"` (leading space) is a *different token* from `"the"`. This is why prompts are sensitive to trailing spaces.

</div>

Every one of these is a **token-boundary** artifact — not a reasoning failure.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** A student says: "GPT is dumb — it can't even count the r's in *strawberry*, and it thinks 9.11 > 9.9." In terms of **tokenization**, explain *precisely* why both happen, and name one tokenizer design change that helps the number case.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

<div class="columns">
<div>

**"strawberry" → count r's.**
The word tokenizes to a few subword chunks (e.g. `straw` + `berry`). The embedding for `berry` is a single vector — the model has **no access to the letters inside it**. Spelling/counting requires character-level information the tokenizer *threw away*.

</div>
<div>

**"9.11 vs 9.9".**
Numbers tokenize **inconsistently**: `11` may be one token, `9` another, boundaries shift with surrounding text. The model learned arithmetic as **token-pattern statistics**, not digit place-value, so it mis-ranks.

</div>
</div>

<div class="keypoint">

**Fix for numbers:** tokenize digits **individually** (Llama-3 does this) — every number becomes a consistent left-to-right digit sequence, so place-value is recoverable. The lesson: many "reasoning" bugs are really **representation** bugs baked in before layer 1.

</div>

---

# Explore it yourself

<div class="notebook">

**🎛 Interactive · BPE merge playground** — type a corpus, press "merge," and watch the most-frequent pair get glued live, merge by merge. *(ES 667 Interactive · `interactive-articles/bpe-merges`)*

</div>

<div class="notebook">

**📓 Notebook · BPE from scratch** — implement the 7-line trainer, train on a small corpus, visualize the merges, then tokenize unseen sentences. *(ES 667 · `notebooks/14-bpe-from-scratch.ipynb` · lab `labs/hw3-nanogpt-tokenization`)*

</div>

<div class="notebook">

**▶ Karpathy · minBPE + "Let's build the GPT Tokenizer"** — a ~2-hour build of byte-level BPE, the clearest walk-through anywhere. *(`github.com/karpathy/minbpe`)*

</div>

---

<!-- _class: section-divider -->

## Part 3 · Pretraining objectives

---

# One lens: hide tokens, predict them

Every objective is the **same move**: hide part of the text, output a **distribution over the missing token(s)**, score it with **cross-entropy** (the NLL of a Categorical, from L1). They differ *only* in **what is hidden** and **what context is allowed**.

![w:600px](figures/lec14/svg/mlm_vs_causal.svg)

<div class="keypoint">

**Causal LM (GPT)** — predict the **next** token, left context only → built to *generate*.
**Masked LM (BERT)** — predict a **masked** token, both sides visible → built to *understand*.
**Span corruption (T5)** — predict a masked **span** with an encoder–decoder → a blend.

</div>

---

# Intuition · comprehension vs composition

An Ng-style analogy for the two objectives, run on the *same* sentence:

<div class="columns">
<div>

**MLM (BERT) = reading comprehension.**
You see the whole passage with one word blanked and fill it in — using words on **both sides**. Ideal for *understanding* a fixed text: classification, retrieval, NER. But you never practice producing the next word with the future hidden.

</div>
<div>

**CLM (GPT) = writing an essay.**
You emit each word seeing only what you have written **so far** — strictly left context. That is literally the *generation* setting: extend the text one token at a time. Weaker per-token representations, but it can actually *write*.

</div>
</div>

<div class="keypoint">

Same loss (cross-entropy on the hidden token), opposite **information available**: two-sided → *understand*, left-only → *generate*. Every other difference in Part 3 is a consequence of this single choice.

</div>

---

# Causal LM (GPT) · predict the next token

Break a sentence into a prediction problem at every position. For $x=[\text{The},\text{cat},\text{sat}]$:

- given `<s>` → predict `The`; given `The` → predict `cat`; given `The cat` → predict `sat`.

$$\mathcal{L}_\text{CLM} = \sum_{t=1}^{T} -\log P_\theta(x_t \mid x_{<t})$$

<div class="insight">

A **causal attention mask** enforces "look backward only." A 2048-token window therefore hands you **2048 supervised problems for free** every step — no labels needed. This is why the objective scales.

</div>

---

# Worked numeric · GPT loss at one step

Context `"The cat …"`, correct next token `sat`. The model emits logits over the vocabulary:
$$\text{logit}(\text{sat})=4.0,\quad \text{logit}(\text{ran})=2.5,\quad \text{logit}(\text{jumped})=2.0,\ \dots$$

Softmax (rest of vocab ≈ negligible): $e^{4.0}\approx54.6$, $e^{2.5}\approx12.2$, $e^{2.0}\approx7.4$.
$$P(\text{sat}\mid\text{The cat}) \approx \frac{54.6}{54.6+12.2+7.4}\approx 0.74 \;\Rightarrow\; \text{loss}=-\log 0.74 \approx \mathbf{0.30}$$

If the model were *sure* ($P=0.99$), loss $\approx 0.01$. Confidently wrong ($P\to0$) → loss $\to\infty$. **Cross-entropy rewards calibrated confidence on the true token.**

---

# Masked LM (BERT) · fill in the blanks

BERT plays a **cloze test**: hide ~15% of tokens, predict them from **both sides**.

1. $x=[\text{The},\text{cat},\text{sat},\text{on},\text{the},\text{mat}]$
2. mask position 4 → $[\text{The},\text{cat},\text{sat},[\text{MASK}],\text{the},\text{mat}]$
3. maximize $P(x_4=\text{on}\mid x_{\setminus 4})$, summed over masked positions:

$$\mathcal{L}_\text{MLM} = \sum_{i\in\text{masked}} -\log P_\theta(x_i \mid x_{\setminus i})$$

![w:440px](figures/lec14/svg/mlm_vs_causal_detail.svg)

No causal mask → **bidirectional** context. Great for classification / retrieval / NER; **bad at generation** — it never practices producing tokens left to right.

---

# Worked numeric · BERT loss for one mask

Input `"… sat [MASK] the …"`, correct answer `on`. Logits at the `[MASK]` position:
$$\text{logit}(\text{on})=3.5,\quad \text{logit}(\text{above})=2.1,\quad \text{logit}(\text{under})=1.5,\ \dots$$

Softmax (rest ≈ negligible): $e^{3.5}\approx33.1$, $e^{2.1}\approx8.2$, $e^{1.5}\approx4.5$.
$$P(\text{on}\mid x_{\setminus4}) \approx \frac{33.1}{33.1+8.2+4.5}\approx 0.72 \;\Rightarrow\; \text{loss}=-\log 0.72 \approx \mathbf{0.33}$$

**Same arithmetic as the GPT step** — softmax then $-\log$ of the true token. Only the *context* changed: BERT used `sat` **and** `the`; GPT would have seen only the left side.

---

# Why mask 15% — and the 80/10/10 split

- Mask **too few** → most sequences give no loss signal → slow training.
- Mask **too many** → too little context survives → the task is impossible.

15% was found empirically. Of the masked positions:

<div class="columns">
<div>

- **80%** → replaced with `[MASK]`
- **10%** → replaced with a **random** token
- **10%** → **left unchanged**

</div>
<div>

The random + unchanged slices stop the model from **cheating** — it can't assume a position is wrong just because it sees `[MASK]`, and it must keep good representations everywhere.

</div>
</div>

<div class="insight">

Mask-then-reconstruct is exactly a **denoising autoencoder** (L19) — BERT denoises corrupted text with a Transformer encoder.

</div>

---

# Span corruption (T5) · text-to-text

T5 frames **every** task as text-in, text-out, and pretrains by masking contiguous **spans**:

```
input :  "Thank you <X> me to your party <Y> week."
target:  "<X> for inviting <Y> last <Z>"
```

<div class="keypoint">

An **encoder** reads the corrupted input (both sides visible, like BERT); a **decoder** generates the missing spans left to right (like GPT). Span corruption is MLM's idea, ported to an encoder–decoder — so one model does translation, summarization, and QA.

</div>

All three objectives are the **same NLL of a Categorical**; only the *masking pattern* changes. Decoder-only (GPT) ultimately won at scale — but the loss never changed.

---

# Which paradigm won — and why

| Era | Front-runner | What it bought |
|:--|:--|:--|
| 2018 | **BERT** (MLM) | cheap, strong embeddings for classification/retrieval |
| 2020 | **GPT-3** (CLM) | scale unlocked few-shot, in-context learning |
| 2023+ | **decoder-only** (CLM) | one model that *generates*, follows instructions, uses tools |

<div class="insight">

**Decoder-only won the LLM race.** BERT still ships inside retrieval pipelines (small, fast, good embeddings); T5-style encoder–decoders survive where structured input→output matters. But for a general assistant, "predict the next token" scaled the furthest — the same objective, just larger.

</div>

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Sentence (6 tokens): `[The] [cat] [sat] [on] [the] [mat]`. The model must predict token 4, **`on`**.

**(a)** Which tokens can a **causal LM (GPT)** condition on to predict `on`?
**(b)** Which tokens can a **masked LM (BERT)** condition on for the same target?
**(c)** In one sentence, why does this make BERT better at *understanding* and GPT better at *generating*?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

<div class="columns">
<div>

**(a) Causal (GPT).**
Left context only: **`The`, `cat`, `sat`** (tokens 1–3). Tokens 5–6 are hidden by the causal mask — the model must predict `on` before it has seen `the mat`.

</div>
<div>

**(b) Masked (BERT).**
Both sides, target excluded: **`The`, `cat`, `sat`, `the`, `mat`** (1–3 and 5–6). It uses the right-hand context `the mat` that GPT is forbidden to see.

</div>
</div>

<div class="keypoint">

**(c)** BERT's two-sided view builds richer **representations** of a fixed sentence (→ understanding/classification), but it never learns to extend text one token at a time. GPT's left-only view is exactly the **generation** setting — predict the future from the past — which is why decoder-only models became the generators.

</div>

---

<!-- _class: section-divider -->

## Part 4 · What pretraining actually learns

---

# One tiny objective, a foundation model

Predicting the hidden token on a trillion-token corpus **forces** the model to acquire, with no labels:

- **Grammar** — closing brackets, agreement, matching tenses
- **Facts** — "The capital of France is ___" needs a fact
- **Reasoning** — "If all A are B and x is A, then x is ___" needs logic
- **Style** — code continuations, formal register, poetry

<div class="keypoint">

Next-token prediction is so rich that a model good at it ends up learning **most of what is in the data** — implicitly. That is the "foundation" in *foundation model*: one self-supervised task, then adapt.

</div>

---

# Intuition · compression *is* understanding

Why should "guess the next token" produce grammar, facts, and reasoning? Because **predicting well requires modeling whatever generated the text.**

<div class="insight">

To put high probability on the next token of a physics paper, a chess game, or a Python function, the model has no shortcut but to internalize physics, chess, and Python. A better next-token predictor is a **better compressor** of the corpus — and to compress the world's text, you must build a model *of the world that wrote it.* (Sutskever's framing; the information-theoretic root of L1: fewer **bits-per-token** = less surprise = more understanding.)

</div>

<div class="keypoint">

**Compression = understanding = low cross-entropy.** The single scalar the optimizer drives down — average $-\log P_\theta(x_t\mid\text{context})$ — is simultaneously the *loss*, the *code length* (L1), and a proxy for *how much the model has understood.*

</div>

<div class="notebook">

**🎛 Interactive · information theory** — watch cross-entropy split into $H(p)+D_\text{KL}(p\Vert q)$; a language model drives its own predictive **bits-per-token** down as it learns. *(Interactive Lab · `interactive/articles/info-theory`)*

</div>

---

# Why the loss is so rich — free supervision at scale

Every position in a context window is its own little labeled example, and the label is just "the next token" — already in the text.

<div class="math-box">

A **1-trillion-token** corpus therefore yields on the order of $10^{12}$ supervised prediction tasks — **for free**, with zero human annotation.

</div>

<div class="insight">

This scale-and-generality combination is *why* next-token prediction — which looks almost trivial — ended up subsuming most of NLP. The architecture barely changed (decoder-only Transformer); **compute and data** did the work.

</div>

---

# Where do the tokens come from?

<div class="columns">
<div>

**Web-scale (the bulk)**
- Common Crawl — filtered web, ~trillions of pages
- C4 / RefinedWeb — quality-filtered web text
- Wikipedia — ~10B tokens, many languages

</div>
<div>

**Specialized (the seasoning)**
- GitHub code (de-licensed)
- ArXiv, books, StackExchange
- **Synthetic** "textbook-quality" data (Phi-3)

</div>
</div>

**Quality beats quantity:** Phi-3 matched a larger web-trained model using far fewer, heavily curated + synthetic tokens. Dedup, language-ID, toxicity and perplexity filtering each buy real benchmark points.

<div class="realworld">

**Pointer to L17.** *How much* compute and money a run costs — the $6ND$ rule, Chinchilla-optimal token budgets, the $-per-training-run economics — is L17's job. Today's takeaway is only: the loss is cheap per token, and there are *a lot* of tokens.

</div>

---

# Worked numeric · same meaning, different bills

A tokenizer trained mostly on English is efficient on English and **wasteful elsewhere**. **Fertility** = tokens per word for the *same* content:

| Text (same meaning) | Words | Tokens | Fertility |
|:--|:--:|:--:|:--:|
| English | 10 | 12 | **1.2** |
| German (compounds) | 9 | 18 | **2.0** |
| Hindi (Devanagari) | 10 | 35 | **3.5** |

<div class="warning">

Higher fertility = more tokens for the *same idea* = more context consumed, more FLOPs, **higher API bill**, and a shorter effective context window. A Hindi user here pays ≈ **3×** what an English user pays to say the same thing.

</div>

<div class="insight">

This is a direct consequence of *what the BPE merges were trained on* (Part 2): English pairs won the vocabulary slots. Multilingual tokenizers (SentencePiece on balanced corpora) narrow — but never fully close — the gap.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4 · fertility.** The *same* 8-word sentence in a low-resource language tokenizes to **20 tokens** under tokenizer A (English-centric) and **12 tokens** under tokenizer B (balanced multilingual).

**(a)** Give the fertility (tokens / word) of each.
**(b)** At \$2 per **1M** tokens, what does each cost to process **100,000** such sentences, and how much does B save?
**(c)** Name one reason a team might still ship the higher-fertility tokenizer A.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

<div class="columns">
<div>

**(a) Fertility = tokens / words.**
$$\text{A}=\tfrac{20}{8}=\mathbf{2.5}\qquad \text{B}=\tfrac{12}{8}=\mathbf{1.5}$$
B packs the same meaning into 40% fewer tokens.

</div>
<div>

**(b) Cost over 100k sentences.**
A: $20\times10^5 = 2.0\text{M} \to \$4.00$
B: $12\times10^5 = 1.2\text{M} \to \$2.40$
**B saves \$1.60** — and every request leaves more room in the context window.

</div>
</div>

<div class="keypoint">

**(c)** A can still win when the workload is **mostly English** (A is the *low*-fertility one there), when you want a mature ecosystem and matching pretrained weights (GPT-2 / Llama), or when a **smaller** vocabulary keeps the embedding table and output softmax cheap. Fertility is one axis of a multi-axis choice — but for non-English users it maps straight to **cost and context**.

</div>

---

# Perplexity · the number everyone quotes

The training loss is average cross-entropy per token; report it in a friendlier unit. **Perplexity** exponentiates it:

$$\text{PPL} = \exp\!\Big(\tfrac{1}{T}\textstyle\sum_t -\log P_\theta(x_t\mid x_{<t})\Big) = 2^{\;\text{bits-per-token}}$$

<div class="insight">

Read it as an **average branching factor**: "at each token the model is as unsure as if choosing uniformly among PPL options." A blind guess over a 50k vocab has PPL 50,000; a good LM on English sits around **10–20**. Lower PPL = fewer bits-per-token = better compression = (previous slide) more understanding.

</div>

<div class="keypoint">

Cross-entropy, bits-per-token, and perplexity are **one number in three costumes** — $\text{bits}=\log_2\text{PPL}=\text{CE}/\ln 2$. **Halving** the per-token loss takes the **square root** of the perplexity — a large win.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5 · perplexity.** Over a 4-token validation snippet, a language model assigns the *true* tokens the probabilities $\tfrac12,\ \tfrac12,\ \tfrac14,\ \tfrac18$.

**(a)** Compute the average cross-entropy in **bits per token**.
**(b)** Convert it to **perplexity**.
**(c)** A second model reports perplexity **1.0** on the same snippet — what did it do, and is that good news?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

**(a) Bits per token** = average of $-\log_2 p$:
$$-\log_2\tfrac12,\ -\log_2\tfrac12,\ -\log_2\tfrac14,\ -\log_2\tfrac18 = 1,1,2,3 \;\Rightarrow\; \frac{1+1+2+3}{4}=\mathbf{1.75\ \text{bits/token}}$$

**(b) Perplexity** $= 2^{\text{bits}} = 2^{1.75}\approx\mathbf{3.36}$ — "as unsure as a uniform choice among ~3.4 tokens."

<div class="keypoint">

**(c)** Perplexity **1.0** means $2^0=$ **0 bits/token** — the model put probability **1** on every true token. On *training* data that is memorization / overfitting; the only way it is good news is on a genuinely held-out set (essentially never at this extreme). Perplexity near 1 is a **leakage alarm**, not a trophy.

</div>

---

# Play with it · from tokens to behavior

<div class="notebook">

**🎛 Interactive · softmax & temperature** — the LM's final layer is a softmax over the vocabulary; slide temperature from peaky (greedy) to flat (creative) and watch the sampling distribution — and its entropy — move. The exact knob behind LLM decoding. *(Interactive Lab · `interactive/articles/softmax-temperature`)*

</div>

<div class="notebook">

**🎛 Interactive · in-context learning** — few-shot prompting is an *emergent* consequence of next-token pretraining; watch a model pick up a pattern from examples in its context with **no weight updates**. *(Interactive Lab · `interactive/articles/in-context-learning`)*

</div>

<div class="notebook">

**🎛 Interactive · BPE merge playground** — close the loop: type a corpus, merge live, then see how the resulting vocabulary changes token counts (and fertility) on new text. *(Interactive Lab · `interactive/articles/bpe-merges`)*

</div>

---

# See it in code

<div class="notebook">

**📓 Notebook · pretraining objectives, side by side** — build the causal mask vs the MLM mask on the same sentence; verify `F.cross_entropy` reproduces the hand-derived GPT and BERT losses from Part 3. *(ES 667 · `labs/hw3-nanogpt-tokenization`)*

</div>

<div class="popquiz">

**Discuss.** A BERT model aces sentence classification but produces mush when you sample from it autoregressively. From Part 3, what *one* property of its training objective explains this — and why is it not a bug?

</div>

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**LLMs see tokens, not letters — BPE builds those tokens by greedily merging the most frequent pair, and pretraining is one cross-entropy loss where GPT, BERT, and T5 differ only in which tokens are hidden.**

</div>

- **Tokenization** — subwords are the compromise between long char sequences and OOV-prone word lists; **byte-level BPE** never chokes.
- **Gotchas** — counting letters, `9.11 > 9.9`, trailing spaces: all **token-boundary** artifacts, not reasoning failures.
- **Pretraining** — CLM (GPT, generate) · MLM (BERT, understand) · span (T5, both) — all NLL of a Categorical.
- **Payoff** — one tiny objective at scale learns grammar, facts, reasoning, and style.

**Next (L17):** scale it — Chinchilla, the $6ND$ rule, and what a trillion tokens actually costs.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from ES 667 course materials and standard references:

- **Tokenization figures, BPE trace, MLM-vs-causal diagrams, hand-traces** — *ES 667 Deep Learning* tokenization materials, N. Batra · `github.com/nipunbatra/dl-teaching`
- **minBPE & "Let's build the GPT Tokenizer"** — A. Karpathy · `github.com/karpathy/minbpe` (the clearest byte-level BPE build)
- **Byte-Pair Encoding for NMT** — Sennrich, Haddow & Birch, ACL 2016
- **BERT (masked LM)** — Devlin, Chang, Lee & Toutanova, NAACL 2019
- **T5 (span corruption, text-to-text)** — Raffel et al., JMLR 2020
- **Pedagogical framing** (crisp, output-first) — A. Ng, *Deep Learning Specialization*
- **"Compression is understanding," next-token as world-modeling** — framing after I. Sutskever (public talks), grounded in the entropy / bits-per-token view of L1
- **Perplexity & bits-per-token** — standard LM evaluation, tracing to Shannon, *A Mathematical Theory of Communication* (1948)
- **Interactive explainers** (BPE merges, softmax & temperature, in-context learning, information theory) — *ES 667 Interactive Lab*, N. Batra

Figures adapted from the ES 667 figure library. Interactive: `interactive/articles/{bpe-merges, softmax-temperature, in-context-learning, info-theory}`. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · byte-level BPE, regex pre-split, SentencePiece

**Byte-level.** Encode text as UTF-8 bytes first, so the base alphabet is exactly the 256 byte values. Any Unicode code point is one or more bytes, so *every* string is representable and `<unk>` disappears entirely — at the cost of a slightly longer base sequence for non-ASCII scripts.

**Regex pre-split.** GPT-2 splits text with a regex (roughly: contractions, letter runs, digit runs, punctuation, whitespace) *before* counting pairs. This forbids merges across those boundaries, so `"New York"` stays two chains and numbers don't glue onto words.

**WordPiece vs SentencePiece.** WordPiece (BERT) chooses the merge that most increases the corpus **likelihood** under a unigram model, not raw frequency. SentencePiece treats whitespace as an ordinary symbol (the `▁` meta-space), so the tokenizer is fully reversible and language-agnostic — the default for multilingual models.

*(Full build: Karpathy `minbpe`; Sennrich et al. 2016 for the original subword formulation.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · all three objectives are one loss

Let $M$ be the set of hidden positions and $c(i)$ the context each objective allows. Every paradigm minimizes

$$\mathcal{L} \;=\; \sum_{i \in M} -\log P_\theta\big(x_i \mid x_{c(i)}\big),$$

the **NLL of a Categorical** (L1). The *only* differences:

| Paradigm | Hidden set $M$ | Context $c(i)$ |
|:--|:--|:--|
| **CLM (GPT)** | every position | strictly-left tokens $x_{<i}$ |
| **MLM (BERT)** | ~15% sampled | all tokens except $M$ |
| **Span (T5)** | contiguous spans | all uncorrupted tokens (encoder) + generated prefix (decoder) |

Because $x$ is effectively one-hot at each target, the cross-entropy equals the **KL divergence to the true next-token distribution** (L1), and $H(p)=0$ — so "minimize cross-entropy," "maximize likelihood," and "match the data distribution" are, once again, **the same objective** viewed three ways.
