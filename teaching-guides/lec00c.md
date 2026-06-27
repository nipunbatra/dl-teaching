# Lecture 0C · Information Theory for ML · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
One idea — **surprise = `−log p`** — generates everything: average it and you get entropy (= the length of the best code); code with the wrong distribution and you get cross-entropy; the gap is KL. The punchline is that training a classifier *is* data compression: cross-entropy loss in bits is literally the compressed size per symbol, and a better predictor is a better compressor.

## Where it sits
Builds on … L00 (every loss is `−log p`, the NLL) · L00B (KL showed up as "extra surprise"; here it gets the full coding story).
Sets up … L01+ (every classifier minimizes `KL(p̂_data ‖ p_θ)`), LLM evaluation (perplexity, bits-per-byte), and the KL terms in VAE/diffusion/RLHF/distillation.

## 80-minute plan
- ⭐ Core (~40 min): surprise `I = −log p` and the three axioms that force the log · bits vs nats · entropy = expected surprise = optimal code length · the weather coding tree (1.5 bits/day) · cross-entropy = coding with the *wrong* `Q` · `H(P,Q) ≥ H(P)` · **KL = the extra bits** = `H(P,Q) − H(P)`. Every student must leave able to read a cross-entropy number as "wasted bits."
- ⭐⭐ Should-cover (~30 min): the classification collapse `CE = KL = NLL` for one-hot labels (worked over 5 examples) · MLE = `min KL(p̂_data ‖ p_θ)` · perplexity = `e^CE` (nats) / `2^CE` (bits) · "a better model is a better compressor" / bits-per-byte · forward vs reverse KL.
- ⭐⭐⭐ Optional / skip-if-tight: the **label-smoothing & temperature** slide (read through entropy) — gist if asked: "label smoothing raises the *target's* entropy so the model is punished for overconfidence (CE gains a floor >0); temperature `T` divides logits before softmax, directly scaling *output* entropy — high T = diverse, low T = greedy." Also the arithmetic-coding aside on fractional bits.
**Do live on the board:** the **two-route KL agreement** (slide "the two routes agree"): `P=(0.7,0.3)`, `Q=(0.5,0.5)`. Route 1: `H(P,Q) = −[0.7·log₂0.5 + 0.3·log₂0.5] = −log₂0.5 = 1.0` bit, floor `H(P) = −0.7log₂0.7 − 0.3log₂0.3 = 0.88` bits, so `KL = 1.0 − 0.88 = 0.12` bits. Route 2: `0.7·log₂(0.7/0.5) + 0.3·log₂(0.3/0.5) = 0.12` bits. It lands because two formulas that *look* unrelated produce the identical number — that's the moment "extra bits" and "average log-ratio" fuse into one concept.

## Teach it like this
Open on pure intuition — "the sun rose" tells you nothing, "it snowed in Gandhinagar" tells you everything — and *derive* `−log p` from three things you'd want surprise to satisfy (depends only on `p`, decreases in `p`, adds for independent events). Make it physical with the weather telegraph: build the optimal code by hand, count 1.5 bits/day, and announce "that number is the entropy." Then break it: forecast wrong (`Q`), reuse the code, count *more* bits — that excess is cross-entropy, and the waste is KL. End on the reframe that pays off all semester: a classifier minimizing cross-entropy is literally compressing the labels toward the truth, and perplexity is that compressed size made tangible.

## Heads-up for YOU (subtle points to get right)  ← most important section
- **Entropy is *expected* (average) surprise, not "the information in a message."** A single outcome carries surprise `−log p(x)`; entropy `H(p) = E[−log p]` averages over the whole distribution. Don't conflate the per-outcome quantity with the average — students who blur them later think "high-probability events have high entropy."
- **The lowest achievable cross-entropy is `H(P)`, NOT zero.** `H(P,Q) ≥ H(P)` with equality iff `Q = P`. CE → 0 *only* when the truth is deterministic (one-hot). If the data is genuinely noisy/ambiguous, the loss floor is `H(P) > 0` — the irreducible uncertainty. Don't tell them "a perfect model has zero loss" in general; that's only true for one-hot targets.
- **`CE = KL` ONLY for one-hot targets.** `H(P,Q) = H(P) + KL(P‖Q)`; the classification loss equals KL *because* the one-hot label has `H = 0`. With label smoothing, `H(target) > 0`, so CE = `H(target) + KL` ≠ KL. Keep the condition attached.
- **Perplexity's base must match the loss's units.** `perplexity = e^CE` if CE is in *nats* (PyTorch's default), `= 2^CE` if CE is in *bits*. Don't write "perplexity = 2^loss" while quoting a PyTorch loss in nats — you'll be off by an exponentiation base. (Sanity check: CE = ln 10 ≈ 2.30 nats ⇒ perplexity = `e^2.30 = 10`.)
- **MLE = min KL uses FORWARD KL** — `KL(p̂_data ‖ p_θ)`, data first. This is *mode-covering*: the model must spread mass wherever the data has mass. Say "forward, mode-covering" explicitly; this is why MLE-trained generative models are safe-but-blurry, and the contrast with reverse KL (mode-seeking) only makes sense if the direction is right.
- **Base of the log doesn't change the optimum.** Switching bits↔nats just multiplies the loss by a constant (`1 nat = 1/ln2 ≈ 1.44 bits`). The `arg min` and the gradients' *direction* are unchanged (a constant scale folds into the learning rate). PyTorch uses natural log because its derivative is cleaner. Don't imply bits vs nats changes *what* the model learns.
- **"Fractional bits" per symbol are real only in aggregate.** You can't send 1.58 bits for one die roll, but arithmetic coding over long sequences achieves the fractional rate *in expectation*. So entropy being `2.58` bits for a die is a long-run average, not a per-symbol packet size.

## Where students stumble (and the fix)
- **Why a *logarithm* specifically.** Hammer the additivity axiom: two independent flips should give *twice* the surprise, but their probability *multiplies*. Log is the unique tool turning "multiply probabilities" into "add information": `−log(p₁p₂) = −log p₁ − log p₂`. That single property is why logs are everywhere in ML losses.
- **Reading the cross-entropy number.** Make "loss = 2.30 nats on a 10-class problem" mean something: that's `ln 10`, i.e. the model is outputting uniform `1/10` — it has learned nothing. Lower loss = fewer bits = better compression.
- **Confusing `H(P)` (the floor) with `H(P,Q)` (what you pay).** Keep them on the board side by side in the weather example: floor 1.5 bits, you-paid 1.75 bits, waste 0.25 bits. The waste is the only part the model controls.
- **Forward/reverse KL direction.** Same fix as L00B: tie each to *what you average over* — forward over the truth (covers everything), reverse over the model (hides in one mode).

## If a student asks…
- "PyTorch uses natural log but bits use base 2 — does that change my loss?" → Only by a constant factor (`ln 2 ≈ 0.693`). The optimum and the gradient direction are identical; the constant folds into the learning rate. Base `e` is chosen for a cleaner derivative (`1/x`).
- "If for one-hot labels CE = KL, why even call it cross-entropy?" → Because the *formula* `−Σ p log q` is cross-entropy regardless of `p`. It coincides with KL only because the one-hot `p` has zero entropy. Use label smoothing and `H(p) > 0`, so it's strictly cross-entropy again.
- "What does training loss = 2.30 mean?" → `ln 10 ≈ 2.3026`. On 10 classes that's a model predicting uniform `1/10` — it has learned nothing yet. (Same number flags the double-softmax bug in L01.)
- "Can cross-entropy loss ever reach zero?" → Only if the truth is deterministic (one-hot) *and* the model is perfect. For data with genuine ambiguity, the floor is the data's own entropy `H(P) > 0`; you can close the gap to it, never beat it.
- "What is bits-per-byte and why do LLM papers report it?" → It's the cross-entropy in bits, normalized per byte of text — literally the compressed file size per byte your model achieves. "A good language model = a good text compressor" is not a metaphor; it's the same number.
- "Compute perplexity for CE = 1.5 nats/token." → `e^1.5 ≈ 4.48` — the model is, on average, as unsure as guessing uniformly among ~4.5 words. Improving from perplexity 20 → 8 means going from "1-in-20 confused" to "1-in-8."
- "Why is KL asymmetric in one line?" → The dice game: you score a candidate die by the surprise it assigns to rolls *from the true die*. Swapping which die generates the rolls changes the score entirely — `KL(P‖Q)` and `KL(Q‖P)` measure different things.

## If you're short on time
Cut: the label-smoothing/temperature slide, the naive-vs-optimal-code slide, and the perplexity *details* (keep one sentence). Never cut: the **two-route KL agreement** (CE = H + KL made arithmetic) and the **`CE = KL = NLL` one-hot collapse** — those are what unify L00, L00B, and today into one number.

## Closing line
"Cross-entropy loss is the number of extra bits your model wastes describing reality — drive it down and you are compressing the world toward the truth, which is the same thing as understanding it."
