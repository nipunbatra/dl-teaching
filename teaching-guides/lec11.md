# Lecture 11 · Seq2Seq & the Motivation for Attention · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
Seq2Seq is two RNNs back to back: an encoder squeezes the whole source into one fixed-size context vector, and a decoder unspools that vector into the target sequence. It works for short sentences and breaks for long ones — the single vector can't hold everything — and that failure is exactly what *demands* attention next lecture.

## Where it sits
Bridge lecture. It takes the RNN/LSTM from L10 and assembles it into the architecture that ran Google Translate 2014–2016, then deliberately walks into its wall (the fixed-length bottleneck) to motivate L12 (attention). Three transferable ideas live here that recur everywhere: **teacher forcing** (training trick), **decoding strategies** (greedy/beam/nucleus — you'll use these for every generative model in the course), and the **bottleneck argument**.

## 80-minute plan
- **(0–8 min) ⭐ Hook.** Translate the 14-word relative-clause sentence into Hindi (different word order). Word-by-word fails; one-vector-then-write struggles; "look back at any source word" is attention (deferred to L12).
- **(8–24 min) ⭐ Encoder-decoder.** Sutskever 2014, BLEU 34; the translator analogy (read whole sentence → mental summary → compose). The PyTorch skeleton: encoder returns `(h, c)`, decoder starts from it. **Do live on the board:** draw the architecture and circle the *single* context vector $c$ — that circle is the whole lecture's villain.
- **(24–40 min) ⭐ Teacher forcing.** The compounding-error problem at training; the training-wheels analogy; the *real* reason is **parallelism** (10–100× speedup), not just safety. Write the loss $-\sum_t\log p(y_t\mid y^*_{<t},x)$.
- **(40–50 min) ⭐⭐ Exposure bias.** The "A dog ran away" cascade. Model trained in a perfect world, tested in a messy one. Mitigation: scheduled sampling, or just scale (Transformers).
- **(50–68 min) ⭐ Decoding.** Greedy fails (the 0.6 vs 0.4 → 0.18 vs 0.32 example); log-probs to avoid underflow; **length normalization** ($1/T^\alpha$, $\alpha\approx0.6$); top-k vs nucleus (the improv-comedian analogy). **Do live on the board:** the $k=2$ beam trace.
- **(68–76 min) ⭐ The bottleneck.** BLEU-vs-length cliff; Sutskever's "reverse the input" hack as a tell-tale sign of an unsolved problem. "Don't use one context vector" → attention.
- **(76–80 min) ⭐⭐⭐ Where Seq2Seq still ships.** Whisper, T5, BART. Skip if tight.

The ⭐⭐⭐ "beam search worked example" slide duplicates the live board example — pick one.

## Teach it like this (hook → sequence → payoff)
**Hook:** the nested Hindi-translation sentence — make them feel that the *whole* nested clause must pass through one vector. **Sequence:** encode→context→decode (the translator's mental summary) → train it efficiently with teacher forcing → pay for that with exposure bias → at inference, decode it well with beam/nucleus → but no matter how you decode, the encoder already lost the beginning of long sentences. **Payoff:** "if the whole sentence must squeeze through one vector, long sentences will always lose" — and the obvious fix is to stop compressing.

## Heads-up for YOU (subtle points to get right)
1. **Teacher forcing is a mandatory engineering reality, not a flaw — and its main payoff is parallelism.** Because every decoder input is known up front (the ground-truth target), all positions train in one batched matmul instead of a sequential loop. Don't present it as "a hack we tolerate"; present it as "the thing that makes training tractable." Exposure bias is the *price*, not a bug.
2. **Exposure bias is a train/inference *distribution* mismatch.** Training: $p(y_t\mid y^*_{<t})$ (true history). Inference: $p(y_t\mid\hat y_{<t})$ (its own, possibly wrong, history). The model never practiced recovering from its own mistakes, so one early error cascades. Name it precisely.
3. **Beam search keeps the *global* top-$k$, not top-$k$ per branch.** At each step you expand all $k$ beams by all $|V|$ tokens → $k\cdot|V|$ candidates → keep the $k$ best *overall*. If you say "keep the best from each beam" you've described something else (and the beam never narrows). The slide's wording ("expand each beam, keep top $k$") means the global top $k$ — say it explicitly.
4. **Why length normalization is needed, stated correctly.** Every $\log p$ is negative, so a raw sum *strictly decreases* with length → beam systematically prefers the shortest valid output. Dividing by $T^\alpha$ ($\alpha\approx0.6$–0.7) makes hypotheses of different lengths *fairly comparable*. The goal is fair comparison, NOT "always prefer long." $\alpha=0$ is broken, $\alpha=1$ is the per-token average.
5. **Search error vs model error.** When output is bad: if a higher-probability sequence exists but the beam missed it, that's *search error* (bigger $k$ helps). If the model itself assigns highest probability to garbage, that's *model error* (no beam width fixes it). Great diagnostic framing students remember.
6. **Top-k is fixed-size; nucleus (top-p) is dynamic.** Nucleus = smallest set whose cumulative probability $\ge p$, so it narrows when the model is confident and widens when uncertain. This is *why* nucleus ($p=0.9$–0.95) is the current default over top-k. Don't describe nucleus as "top-k with a different number."

## Where students stumble (and the fix)
- **"How does the decoder know when to stop?"** A special `<EOS>` token is in the vocabulary; generation stops when the decoder emits it. (And it starts from `<BOS>`/`<s>`.) Sounds trivial; gets asked every year.
- **Confusing greedy with beam.** Greedy = one hiker taking the steepest step every fork (commits immediately, $k=1$). Beam = $k$ hikers exploring forks in parallel, ranked globally. Use the numeric: greedy commits to "A" (0.6 > 0.4) and ends worse; beam keeps "The" alive and wins.
- **Thinking the bottleneck is "the vector is too small, make it bigger."** It's *information-theoretic*: for any fixed dimension there is a sentence that exceeds it. Bigger vector delays the cliff; it doesn't remove it. (Tie to L12's "fundamental, not engineering" line.)

## If a student asks…
- **"Why not score all possible output sequences and pick the best?"** The space is $|V|^T$ — e.g. $50000^{20}$. Intractable. Beam search is a heuristic that keeps only $k$ partial hypotheses alive.
- **"Does teacher forcing mean the model never learns to fix its own mistakes?"** (hard) Essentially yes — it only ever sees ground-truth history, so it never practices recovery. That's exactly exposure bias. Mitigations: scheduled sampling (sometimes feed the model's own prediction during training), or sidestep it with massive-scale pretraining.
- **"Why take logs?"** Multiplying 20 small probabilities underflows to 0 in floating point. $\log(ab)=\log a+\log b$ turns the product into a numerically stable sum. Beam search compares *sums of log-probs*.
- **"Why does greedy sometimes loop, e.g. 'I I I I…'?"** Greedy gets stuck in a high-probability repetition attractor — each repeated token is locally most likely. Nucleus/top-p sampling plus temperature breaks the loop by injecting controlled randomness.
- **"Why did reversing the input help (Sutskever's trick)?"** It puts the *last* source words (now first) closest to where the decoder starts, shortening the path that information must travel through the hidden-state chain. It's a hack that relocates the leakage rather than removing it — a giant hint that attention was needed.
- **"Shared vs separate vocabulary?"** Separate = each language its own embedding matrix (can specialize). Shared = one matrix (saves params, "Paris" is the same token in both). Modern multilingual models share via SentencePiece across 100+ languages.

## If you're short on time
- **Cut:** the "inside `nn.LSTM`" I/O slide; the ⭐⭐⭐ beam worked-example slide (do it on the board instead); the "where Seq2Seq still ships" applications slide.
- **Never cut:** teacher forcing (with the parallelism reason), the greedy-fails numeric, length normalization, and the bottleneck argument. Those four are what L12 builds on.

## Live board example (≈5 min): beam search with $k=2$ beats greedy
Vocab toy. Step 1 candidates (log-probs): **A** = −0.5, **The** = −0.7. Keep both.
Step 2, expand and keep the global top 2:

| sequence | log-prob |
|---|---|
| `The cat` | −0.7 + (−0.6) = **−1.3** |
| `A dog`   | −0.5 + (−0.9) = **−1.4** |
| `A feline`| −0.5 + (−1.1) = −1.6 |
| `The dog` | −0.7 + (−1.3) = −2.0 |

Survivors: `The cat` (−1.3) and `A dog` (−1.4).
**The aha:** greedy commits to "A" at step 1 because −0.5 > −0.7, and lands on the worse path (−1.4). Beam kept the *locally worse* "The…" alive and found the globally better "The cat" (−1.3). Then say one line: at the end, divide each survivor's score by $T^{0.6}$ so a 3-word and a 5-word hypothesis compete fairly.

## Closing line
*"If the whole sentence must squeeze through one vector, long sentences will always lose. Next lecture: stop compressing — let the decoder look back at every word."*
