# Lecture 14 · Tokenization & Pretraining Paradigms · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
Tokenization is the first design decision of an LLM: BPE greedily merges the most frequent adjacent symbol-pairs until you have a 32k–128k subword vocabulary that balances short sequences (efficient) against zero out-of-vocabulary (robust). Pretraining then optimizes one of three negative-log-likelihood objectives over those tokens — MLM (BERT, understanding), CLM (GPT, generation), or span corruption (T5, both) — and the decoder-only CLM recipe won the scaled-LLM race.

## Where it sits
The "inputs and objectives" lecture that turns the Transformer (L13) into a real language model. It explains *what the model actually sees* (tokens, not letters — the root of half of all LLM bugs), the three training objectives that define BERT/GPT/T5, and the data + compute economics ($6ND$, ~$5M for 70B) that set up L15 (LLMs / Chinchilla scaling). It closes the arc: attention + scale really was the thing.

## 80-minute plan
- **(0–8 min) ⭐ Hook.** "How many tokens is 'GPT-4 is great!'?" Four options; answer is (d) *it depends on the tokenizer*. Promise this also explains "count the r's in strawberry."
- **(8–22 min) ⭐ Why tokenization is hard.** The three failed alternatives table (chars → $O(N^2)$ blowup; words → huge vocab + OOV; morphemes → language-specific). Subword is the Goldilocks zone. The 36× character-LM cost example.
- **(22–40 min) ⭐ BPE step by step.** The 7-line algorithm; data-compression analogy ("ABCABC" → Z). **Do live on the board:** the "low lower newest widest" merge trace. Then byte-level BPE (GPT-2): start from 256 bytes → zero OOV; pretokenize by regex; the fertility tax on non-English.
- **(40–50 min) ⭐ Tokenization gotchas.** strawberry, arithmetic, leading-space tokens. Tie each back to "the model sees token IDs, not characters."
- **(50–68 min) ⭐ Three pretraining paradigms.** BERT (cloze test, MLM, bidirectional, encoder-only); GPT (predictive-keyboard, CLM, causal, decoder-only); T5 (text-to-text, span corruption, enc-dec). The MLM and CLM loss equations + the one-mask worked numerics.
- **(68–76 min) ⭐⭐ Why CLM won + data quality.** Next-token prediction is $10^{12}$ free supervised tasks; Phi-3 (3T curated) ≈ Llama-2 (2T web); curation is the hot area.
- **(76–80 min) ⭐⭐⭐ Compute economics.** $6ND$; ~$5M for a 70B run. Skip the derivation if tight, keep the number.

The ⭐⭐⭐ "where the 6 comes from" derivation is the cut — keep "$\text{FLOPs}\approx6ND$."

## Teach it like this (hook → sequence → payoff)
**Hook:** the token-count quiz — let them argue, then reveal it's a property of the *tokenizer*, not the sentence. **Sequence:** why every naive tokenization fails → BPE finds the sweet spot by greedy frequency merging (trace it live) → byte-level BPE kills OOV → but the model now sees chunks, which causes the famous bugs → with tokens fixed, the three objectives (MLM/CLM/span) are just *which tokens you hide* → CLM at scale subsumed NLP → here's what it costs. **Payoff:** "LLMs see tokens, not letters — half their bugs start there," and all three paradigms are the *same* NLL-of-a-Categorical from L00, differing only in masking.

## Heads-up for YOU (subtle points to get right)
1. **"Count the r's in strawberry" is mechanical, not a reasoning failure.** "strawberry" tokenizes to chunks like `["straw","berry"]` → the model's input is two token IDs. It *literally never sees the letter 'r'* as a unit; to answer it must recall memorized spelling associations of those IDs. Frame it as an input-representation limitation. Same root cause for arithmetic (e.g. "1234" may be one token but "12345" splits differently, so the model memorizes token patterns instead of manipulating digits) and for trailing-space sensitivity (" the" and "the" are *different tokens*).
2. **BPE is a *greedy* heuristic, not an optimal compressor.** It merges the most frequent pair at each step; it does not search for a globally optimal vocabulary. Fast, deterministic at inference (apply merges in the learned order), good enough — but a sharp student asking "is it optimal?" should hear "no, greedy."
3. **Don't conflate objective with architecture.** MLM (objective) is typically paired with an encoder-only architecture (BERT, sees both directions, no causal mask). CLM (objective) is paired with decoder-only (GPT, causal mask). T5's span-corruption pairs with enc-dec. Students mix "BERT vs GPT" (models) with "MLM vs CLM" (losses) — separate the two axes explicitly.
4. **Why BERT needs a `[MASK]` token instead of deleting the word.** Deleting shifts the sequence and destroys positional alignment — the model wouldn't know *where* the gap is. `[MASK]` is a placeholder that preserves positions. (And the 80/10/10 split — 80% `[MASK]`, 10% random token, 10% unchanged — exists so the model can't cheat: at fine-tune/inference time there's no `[MASK]`, so it must build good representations of *every* position, not just masked ones.)
5. **The $6ND$ rule is a math-operations heuristic, not wall-clock or exact cycles.** Per token: forward ≈ $2N$ FLOPs (dominated by the FFN's two matmuls; the "2" is one multiply + one add per parameter), backward ≈ $2\times$ forward = $4N$, total $6N$; times $D$ tokens → $6ND$. Real GPU time depends on memory bandwidth and model-FLOPs-utilization (MFU, often 30–50%). Sanity check it live: 70B × 1.4T × 6 ≈ $5.9\times10^{23}$ FLOPs ≈ ~$5M.
6. **Byte-level BPE guarantees 0% OOV.** Because every string decomposes into the 256 possible byte values, *any* input — emoji, foreign script, binary garbage — is tokenizable with no `<unk>`. That's the GPT-2 breakthrough, and why it's the modern default. The cost: a vocab trained mostly on English has high *fertility* on other languages (more tokens per word → more context and cost for non-English users).

## Where students stumble (and the fix)
- **"Why not just use characters?"** ~5 chars per word → sequences ~5× longer → attention is $O(N^2)$ → ~25× the compute, and the model wastes capacity relearning that "t-h-e" is a word. Subword keeps common words as single tokens and only splits rare ones.
- **Thinking BERT could generate text if you just sampled from it.** It was never trained to produce tokens left-to-right; sampling autoregressively from a masked LM yields mush. MLM = understanding/embeddings; CLM = generation. (At scale GPT-3+ closed the *understanding* gap from the other direction, which is why decoder-only won outright.)
- **Believing token count is a property of the sentence.** It's a property of the *tokenizer*. Same text, different model → different count → different context cost. Show the strawberry token IDs to make it visceral.

## If a student asks…
- **"Is BPE guaranteed to find the optimal compression?"** (hard) No — it's greedy. Each step merges the locally most frequent pair; the result is fast and effective but not globally optimal. (Contrast: WordPiece uses a likelihood-based merge criterion; SentencePiece treats whitespace as a normal character so it's language-agnostic.)
- **"If BERT understands so well, why did GPT win?"** BERT can't generate (bidirectional training, no future at inference). GPT's left-to-right objective makes it a native generator, and next-token prediction at scale turned out to subsume understanding too. Decoder-only became the dominant paradigm.
- **"Why is next-token prediction such a rich objective?"** It forces syntax, semantics, world knowledge ("capital of France is…"), reasoning ("if all A are B…"), and style — all to lower one loss. A 1T-token corpus is $\sim10^{12}$ self-supervised tasks with zero human labeling.
- **"Why does data quality beat quantity (Phi-3)?"** Phi-3's 3T heavily-filtered/synthetic "textbook-quality" tokens matched Llama-2 7B's 2T web tokens. Curation (dedup, language-ID, toxicity/repetition/perplexity filters, synthetic augmentation) each buys 1–5 benchmark points — currently the hottest research area.
- **"Why does a leading space change the answer?"** " the" and "the" are different token IDs. Prompts are sensitive to trailing/leading spaces because they change the tokenization, hence the model's input.
- **"How much does it cost to train a frontier model?"** $6ND$: 70B on 1.4T tokens ≈ $5.9\times10^{23}$ FLOPs ≈ ~11 days on 4k A100s ≈ ~$5M for one run; Llama-3 70B reportedly ~$80M with experiments; GPT-4 class ~$100M+. A decade ago a net cost tens of dollars.

## If you're short on time
- **Cut:** the ⭐⭐⭐ $6ND$ derivation (keep the formula + the one number); the second BPE example ("hug bug rug"); the pretraining-data-sources and copyright slides; the scaling-recap table (or show it for 30 sec as "architecture barely changed, scale did").
- **Never cut:** the BPE merge trace on the board, the strawberry/arithmetic/space gotchas, and the BERT-vs-GPT-vs-T5 = MLM/CLM/span = "which tokens you hide" framing. Those are the conceptual core.

## Live board example (≈5 min): BPE merge trace
Corpus: `low low low lower newest widest`. Split to characters with end-marker `</w>`:
`l o w </w>`, `l o w e r </w>`, `n e w e s t </w>`, `w i d e s t </w>`.
- **Merge 1:** several pairs tie at count 2 (`l o`, `o w`, `e s`, `s t`, …); break the tie — pick **`(e,s)→es`**. Now `...n e w es t...`, `...w i d es t...`.
- **Merge 2:** `(es,t)` now appears twice → **`(es,t)→est`**.
- **Merge 3:** `(l,o)→lo` (count 2).
- **Merge 4:** `(lo,w)→low` (count 2). Result: `low`, `low e r`, `n e w est`, `w i d est`.

**The aha:** with *zero* linguistic knowledge — pure greedy frequency counting — BPE discovered a whole common word (`low`) *and* a reusable grammatical suffix (`est`). Then close the loop: at inference, apply merges 1–4 in order to a *new* word `lowest` → `low est` — never seen in training, yet tokenized into two familiar units. That's why subword has no OOV.

## Closing line
*"LLMs see tokens, not letters — half their bugs start there. Next: scale it — Chinchilla, RoPE, GQA, and what a trillion tokens buys you."*
