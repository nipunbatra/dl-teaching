# Lecture 12 · The Attention Mechanism · Teaching Guide
*First-time-instructor companion. Audience: undergrads, one ML course.*

## The spine (say this in 2 sentences)
Attention is a differentiable dictionary lookup: a query is compared against keys to produce softmax weights, and the output is a weighted average of the corresponding values. Because every output position can look directly at every input position, the fixed-vector bottleneck of L11 simply disappears.

## Where it sits
The hinge of the whole sequence. L11 ended on the bottleneck; this lecture removes it. It's also the single most influential idea in deep learning between backprop and diffusion. Everything in L13 (the Transformer is stacked self-attention), L14, and every modern LLM is built on the QKV mechanism introduced here. If students leave with one formula memorized, it should be $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$.

## 80-minute plan
- **(0–8 min) ⭐ Recap the cliff + the one-line fix.** "Don't read from one fixed vector — peek at every encoder state and decide which matter." Bahdanau 2014.
- **(8–18 min) ⭐ Attention as soft alignment.** The translation heatmap: rough diagonal, "traversé"↔"cross", "it"↔"animal" coreference emerging for free. **Do live on the board:** the 3-source-word toy ("the cat slept" → context vector) so they see a real weighted mixture before any QKV.
- **(18–32 min) ⭐ From Bahdanau to Luong.** Additive (learned MLP score) vs multiplicative (dot product). Why multiplicative won: GPU-friendly + the learned projections absorb the MLP's job.
- **(32–52 min) ⭐ Q, K, V.** Library analogy (query=question, key=spine title, value=content). Why K and V are *separate* roles. The Python-dict mental model (hard vs soft retrieval). **Do live on the board:** the scaled dot-product 2-token example.
- **(52–64 min) ⭐ Why $\sqrt{d_k}$.** The grading-on-a-curve analogy; the variance argument ($\mathrm{Var}(Q\cdot K)=d_k$, so scores $\sim\pm\sqrt{d_k}$); the $\pm22$-at-$d_k{=}512$ → one-hot → dead gradient story.
- **(64–74 min) ⭐ Self- vs cross-attention + causal mask.** "bank/river" disambiguation = self-attention; decoder reading encoder = cross-attention. The $-\infty$ mask makes future weights exactly 0.
- **(74–80 min) ⭐⭐ The $O(n^2)$ wall + what attention unlocked.** Double context → 4× cost; teases FlashAttention/KV-cache. Skip the wall detail if tight.

The two ⭐⭐⭐ slides (variance derivation; you can fold its punchline into the $\sqrt{d_k}$ section) are the cuts.

## Teach it like this (hook → sequence → payoff)
**Hook:** the Google-search pop quiz — query vs page-titles vs page-contents, and "does it return one page or blend the top few?" Hold the answer. **Sequence:** show a real attention heatmap (it *looks* like alignment) → score it with a dot product → name the three roles Q/K/V → fix the softmax saturation with $\sqrt{d_k}$ → generalize to self-attention (the sentence reads itself) → mask the future for generation. **Payoff:** the search-engine intuition they had *before* any math literally *is* the formula $\mathrm{softmax}(QK^\top/\sqrt{d_k})V$ — answer (c), a weighted blend of values. That's why attention spread so fast.

## Heads-up for YOU (subtle points to get right)
1. **$\sqrt{d_k}$ comes from a variance argument — get it exact.** If $Q$ and $K$ have i.i.d. zero-mean, unit-variance entries, then $Q\cdot K=\sum_{i=1}^{d_k}q_ik_i$ has variance $d_k$ (sum of $d_k$ independent unit-variance terms), so standard deviation $\sqrt{d_k}$. At $d_k=512$ raw scores are $\sim\pm22$; softmax of $[22,-22,\dots]$ is essentially one-hot → gradient $\approx0$ → learning stalls. Dividing by $\sqrt{d_k}$ restores variance to 1. It's a temperature derived from first principles, not tuned. Bonus: this makes the block *dimension-invariant* — works at $d_k=64$ or 4096 with no retuning.
2. **K and V are separate on purpose: "what I am" vs "what I contribute."** The key is the label used to *match* against queries; the value is the *payload* blended into the output. A word like "bank" should be *found* by the query "financial" (its key) but *contribute* its full contextual embedding (its value). Early models (Luong) tied K=V; separating them roughly doubles a head's expressiveness. Don't let students think K and V are "the same thing twice."
3. **Attention is permutation-EQUIVARIANT, not invariant — say it precisely.** Permute the input tokens and the *output rows permute identically* (equivariant). Invariant would mean the output doesn't change at all (like sum-pooling). This is the precise reason positional encoding is needed in L13. Using "invariant" here is a common and confusing slip.
4. **The causal mask replaces scores with $-\infty$ *before* softmax — it doesn't delete tokens.** Because $e^{-\infty}=0$, future positions get *exactly* 0 weight after softmax and the rows still sum to 1. The mask is the *only* difference between BERT-style (bidirectional) and GPT-style (causal) attention — same module, different mask. Watch for the fully-masked-row NaN (softmax over all $-\infty$).
5. **"Multiplicative won" needs the right reason.** Two reasons: (i) a dot product is one matmul GPUs love, vs Bahdanau's extra $W_1,W_2\in\mathbb R^{d\times d}$, vector $v$, and a $\tanh$ pass; (ii) once you add learned $W_Q,W_K$ projections, the model can *learn* the similarity in projection space — you no longer need the MLP to learn it. Don't say "they're identical"; say "with learned projections, dot-product is enough."
6. **Attention weights are suggestive, not explanations.** Heatmaps are interpretable and were a landmark, but a head can attend to tokens that don't causally drive the output. Hedge this when you sell "interpretability."

## Where students stumble (and the fix)
- **"Where do Q, K, V come from?"** They're three learned *linear projections of the same input(s)*. Cross-attention: Q from the decoder, K & V from the encoder. Self-attention: all three from the same sequence ($Q=XW_Q$, etc.). Do the projection numeric (input $[1,2,3,4]$ → q $[4,5]$, k $[6,5]$, v $[5,4]$) so "one actor, three costumes" lands.
- **"Isn't this just a Google search?"** Almost — but that's *hard* retrieval (return the one best page). Attention is *soft* retrieval (a weighted blend of all values). The $\sqrt{d_k}$ scaling is precisely what keeps it soft instead of collapsing to hard top-1.
- **"How does self-attention know word order if it sees the whole set at once?"** It doesn't — that's the equivariance flaw, fixed by positional encoding next lecture. Demo: shuffle the input tokens; outputs just permute, content unchanged.

## If a student asks…
- **"If self-attention is permutation-equivariant, doesn't it ignore order entirely?"** (hard) Yes — by itself it treats the input as a *set*. Two sentences with the same words in different order produce the same (up to permutation) representations. L13's positional encoding injects order to break this.
- **"Why softmax and not just normalize the scores?"** Softmax turns arbitrary real scores into a non-negative distribution summing to 1 (stable output magnitude) and is smooth/differentiable. It also gives the sharp-vs-soft control that the $\sqrt{d_k}$ scaling tunes.
- **"What's the actual difference between BERT and GPT attention?"** One line of code: the causal mask. BERT sees both directions (no mask); GPT masks the future (lower-triangular).
- **"Why does doubling the context cost 4×?"** $QK^\top$ is an $n\times n$ matrix → $O(n^2)$ memory and $O(n^2 d)$ compute. At $n=8192$ that's ~67M entries per head per layer (~134 MB fp16). This is the wall behind FlashAttention and KV-caching (L23).
- **"Multi-head — does separating K and V double parameters?"** It doubles one *head's* projection params but roughly doubles its expressiveness too; total block params stay $4d_{\text{model}}^2$ regardless of head count (you partition $d_{\text{model}}$, you don't add to it). Full count is an L13 topic.

## If you're short on time
- **Cut:** the ⭐⭐⭐ variance-derivation slide (state the punchline only); the Bahdanau term-by-term numeric; the self-attention-vs-convolution slide; the "four kinds of attention" table.
- **Never cut:** the 3×3 / 2-token scaled dot-product worked by hand, the $\sqrt{d_k}$ intuition, the Q/K/V library analogy, and the causal-mask $-\infty$ trick. Those carry directly into L13.

## Live board example (≈5 min): scaled dot-product, 2 tokens
$d_k=2$, so $\sqrt{d_k}\approx1.414$. Query $Q_1=[1,0]$. Keys $K_1=[1,1]$, $K_2=[2,0]$. Values $V_1=[0.1,0.2]$, $V_2=[0.3,0.4]$.
1. **Scores:** $Q_1\!\cdot\!K_1=1$, $Q_1\!\cdot\!K_2=2$ (token 1's query matches $K_2$ better).
2. **Scale:** $[1,2]/\sqrt2=[0.707,1.414]$.
3. **Softmax:** $[0.33,0.67]$.
4. **Output:** $0.33\,V_1+0.67\,V_2=[0.234,0.334]$.
**The aha:** the output is 67% the payload of $V_2$ and 33% of $V_1$ — the query *pulled* mostly from the better-matching value. That is differentiable lookup. Optional 30-second extension: show that *without* the $/\sqrt{d_k}$, if scores were $[10,20]$ the softmax would be ≈$[0.00005,0.99995]$ — one-hot, dead gradient. Same shape, scaling is the whole point.

## Closing line
*"Attention is a differentiable dictionary lookup. Next: stack it, multi-head it, mask it, norm it — and you've built the Transformer."*
