---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Attention & Transformers I · The Attention Mechanism

## Lecture 14 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

L13 left the decoder starving behind one fixed vector. Today we tear down that wall with a single move: **let each position build its output as a weighted sum of value vectors** — and let the weights be chosen, softly and differentiably, by how well a **query** matches every **key**.

<div class="keypoint">

**Attention is a soft, differentiable dictionary lookup: `output = Σᵢ softmaxᵢ(query · keyᵢ) · valueᵢ`.**
Everything else — three learned projections (Q/K/V), the $\sqrt{d_k}$, self vs cross, the causal mask — is detail bolted onto that one line.

</div>

You met the fixed-context **bottleneck** in L13. Today we answer the question it forced open: **what should the decoder read, if not one squeezed vector?**

$$\underbrace{\text{weighted sum of values}}_{\text{the whole idea}} \;\rightarrow\; \underbrace{Q,K,V \text{ projections}}_{\text{how scores are learned}} \;\rightarrow\; \underbrace{\div\sqrt{d_k}}_{\text{keep softmax soft}} \;\rightarrow\; \underbrace{\text{self / cross / causal mask}}_{\text{who reads whom}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**You'll reuse today's ideas in:**
- **L15** — assemble this mechanism into the **Transformer** (multi-head, residual + LayerNorm, positions). *L15 will not re-derive attention — it is built here.*
- **L16–L18** — the causal mask you write today *is* the GPT decoder
- **L23** — the $O(n^2)$ cost we hit at the end is what FlashAttention & KV-caching attack

</div>
<div>

**Borrowed from earlier lectures** (we build on, not repeat):
- the **bottleneck** that makes attention inevitable — **L13**
- the **softmax** + cross-entropy machinery — **L1**
- **cosine / dot-product** similarity of embeddings — **L13**

</div>
</div>

<div class="insight">

One thread ties it together: the dot-product-then-softmax you used to *compare embeddings* in L13, now turned into a **read operation** — score every source position, then blend their contents.

</div>

---

<!-- _class: section-divider -->

## Part 1 · From the bottleneck to a weighted sum of values

---

# The one-line fix: stop compressing, start pulling

L13 forced the *entire* source through one context vector $\mathbf c$ — the encoder **pushed** a lossy summary and the decoder took whatever fit. Flip the direction.

<div class="columns">
<div>

### Old · push mode
Encoder compresses everything into $\mathbf c$; the decoder reads only $\mathbf c$.
- one-shot, mandatory compression
- long sentences overflow → BLEU cliff

</div>
<div>

### New · pull mode
Keep **every** encoder state around. At each step the decoder **pulls** what it needs.
- no compression forced
- decoder chooses relevance *per output token*

</div>
</div>

<div class="keypoint">

Attention is the mechanism for the *pull* — a differentiable "look up the source words I need **right now**." Bahdanau et al. (2015) launched a decade of NLP with exactly this idea.

</div>

---

# The whole idea, stated before any algebra

<div class="keypoint">

**The output is a weighted sum of value vectors.** The weights are a softmax over compatibility scores between one **query** and every **key**. That is all attention is.

</div>

| Piece | In one phrase |
|:--|:--|
| **Query** $q$ | what this position is looking for |
| **Key** $k_i$ | what each input advertises it offers |
| **Value** $v_i$ | the content each input actually carries |
| **Score** | $q\cdot k_i$ — how well key $i$ matches the query |
| **Weight** | $\text{softmax}_i(q\cdot k_i)$ — a distribution summing to 1 |
| **Output** | $\sum_i \text{weight}_i\, v_i$ — the weighted blend of values |

$$\text{output} \;=\; \sum_i \underbrace{\text{softmax}_i(q\cdot k_i)}_{\text{weights}}\; v_i$$

*Q/K/V projections, the $\sqrt{d_k}$, and masks are detail built on top of this one line over the next four parts.*

---

# Intuition · attention is a *learned* weighted average

Strip attention to its skeleton and it is the most familiar object in statistics — an **average of the value vectors**:

$$\text{output} = \sum_i w_i\, v_i,\qquad \sum_i w_i = 1,\ \ w_i\ge 0.$$

<div class="columns">
<div>

**Plain pooling** uses *fixed, equal* weights $w_i=\tfrac1n$ — every token counts the same regardless of input. It cannot tell *"loan"* from *"the."*

</div>
<div>

**Attention** makes the weights $w_i=\text{softmax}_i(q\cdot k_i)$ — **chosen per query, from the content itself.** Relevant tokens get more mass; the rest fade.

</div>
</div>

<div class="insight">

That single upgrade — *fixed* weights → *input-dependent, learned* weights — is the whole leap. Attention is a weighted average whose weights are **a function of the data**, recomputed fresh for every token and differentiable end-to-end. Everything after this is *how* those weights get chosen.

</div>

---

# What attention looks like — literally, a heatmap

![w:720px](figures/lec12/svg/attention_heatmap.svg)

Each **target** word (row) computes a **distribution over source words** (columns). Darker cell = more attention mass placed there. This picture *is* the softmax weights, one row per decoding step.

---

# Reading the heatmap: alignment emerges for free

Three things to notice — none of them were programmed in:

1. **A rough diagonal** — most translation alignments are monotonic, and the model discovers that.
2. **"traversé" attends to "cross"** — semantic alignment *across a language barrier*.
3. **"it" attends to both "it" and "animal"** — coreference, learned purely from translation loss.

<div class="insight">

Nobody told the model what "it" refers to. It learned this by minimizing loss. Attention made linguistic structure **visible** for the first time — a caveat for L18: the weights are *suggestive*, not a full causal explanation.

</div>

---

# Attention by hand, before any Q/K/V

Source · *"the cat slept"*.  Target step · the decoder is about to emit French *"chat"* (cat).

<div class="math-box">

Encoder states (toy 2-d): $\;h_1=[1,0]$ (the), $\;h_2=[0,1]$ (cat), $\;h_3=[0.2,0.1]$ (slept)
Decoder state (the query): $\;s=[0.1,0.9]$

Raw scores $s\cdot h_i$: $\quad 0.1,\quad 0.9,\quad 0.11$
Softmax: $\quad [0.24,\ 0.53,\ 0.24]$ — *"cat"* wins the most weight.
Context read: $\;c = 0.24\,h_1 + 0.53\,h_2 + 0.24\,h_3 \approx [0.29,\ 0.55]$

</div>

The decoder reads a **weighted mixture**, dominated by the relevant source word — *not* the squeezed $\mathbf c$ of L13. That is one step of attention, before we've named a single projection.

---

# See it, play with it

<div class="notebook">

**🎛 Interactive · attention heatmap** — click a target token and watch its attention weights light up over the source in real time; hover to read the exact softmax mass on each source word. *(Interactive Lab · `interactive/attention`)*

</div>

<div class="notebook">

**📓 Notebook · attention by hand** — build one attention head from three `nn.Linear` layers, compute scores → softmax → weighted values on a toy sentence, and visualize the resulting heatmap. *(ES 667 · `notebooks/12-attention-by-hand.ipynb`)*

</div>

<div class="notebook">

**🎛 Interactive · the bottleneck you're escaping** — revisit L13's seq2seq compression: watch translation quality fall off a cliff as the sentence grows and *everything* is forced through one context vector. Attention is the fix you're building on top. *(Interactive Lab · `interactive/seq2seq-bottleneck`)*

</div>

---

<!-- _class: section-divider -->

## Part 2 · Query / Key / Value — three learned projections

---

# The library analogy: three roles, not one

<div class="keypoint">

You walk into a library with a question.
- **Query** · *"I need info on the 2008 financial crisis."*
- **Key** · the title on each book's spine (*"The 2008 Meltdown", "The Great Depression"…*).
- **Value** · the actual *content* inside each book.

You match your **query** against the **keys**; the match scores decide how much of each book's **value** you blend into your answer.

</div>

That is attention in one breath: a soft library lookup. The next slides make it math — but the analogy is the whole intuition.

---

# Soft retrieval: why keys and values are separate

![w:840px](figures/lec12/svg/attention_soft_retrieval.svg)

<div class="keypoint">

**Keys say "what I am"; values say "what I contribute."** A word like *"bank"* should be *found* by the query *"financial"* (its key), but *contribute* its full contextual meaning (its value). Two roles → two projections → roughly double the expressiveness of tying them.

</div>

---

# Attention is differentiable `db[query]`

```python
# Hard retrieval — a Python dict
db  = {k1: v1, k2: v2, k3: v3}
out = db[query]                       # exact match → exactly one value

# Soft retrieval — attention
scores  = [sim(query, k) for k in (k1, k2, k3)]   # similarities
weights = softmax(scores)                          # a distribution
out     = weights[0]*v1 + weights[1]*v2 + weights[2]*v3
```

<div class="insight">

Hard lookup returns **one** value; attention returns a **softmax-weighted blend** of all values. Making the lookup soft is exactly what makes it *differentiable* — gradients can flow back and reshape what "sim", keys, and values mean. Everything else is `db[query]` with the edges rounded off.

</div>

---

# Q, K, V are three learned projections of one input

![w:900px](figures/lec12/svg/qkv_attention.svg)

Every token embedding $x$ is projected three ways by learned matrices: $\;q = x^\top W_Q,\; k = x^\top W_K,\; v = x^\top W_V$. Stack over all tokens → matrices $Q, K, V$. The network *learns* the three "costumes" that make the lookup work.

---

# Why project at all? — the freedom Q/K/V buy

Why not just set $Q=K=V=X$ and skip the matrices? Because the projections give the network three things it otherwise couldn't have:

<div class="columns">
<div>

- **A learned similarity.** $W_Q, W_K$ define *what "relevant" means* — the dot product $q\cdot k$ can align words that are far apart in the raw embedding space.
- **A separate content channel.** $W_V$ lets a token *contribute* something different from what it is *matched on* (keys ≠ values).

</div>
<div>

- **Dimension freedom.** $Q,K$ can live in a smaller $d_k$ than the model width $d_\text{model}$ — cheaper scores, and exactly what lets us split the width across several **heads** in L15.

</div>
</div>

<div class="insight">

Raw embeddings serve *many* jobs at once; the three projections carve out task-specific query, key, and value subspaces. Skip them and every head would be forced to match and carry with the same vector.

</div>

---

# One actor, three roles — the projection, worked

Input vector for *"cat"* · $x = [1,2,3,4]^\top$ ($d=4$). Project to $d_k=2$ with three $4\times2$ matrices:

$$W_Q = \begin{pmatrix} 1&0\\0&1\\1&1\\0&0 \end{pmatrix},\quad W_K = \begin{pmatrix} 0&1\\1&0\\0&0\\1&1 \end{pmatrix},\quad W_V = \begin{pmatrix} 1&1\\0&0\\0&1\\1&0 \end{pmatrix}$$

- $q = x^\top W_Q = [\,1{+}3,\ 2{+}3\,] = [4,\ 5]$
- $k = x^\top W_K = [\,2{+}4,\ 1{+}4\,] = [6,\ 5]$
- $v = x^\top W_V = [\,1{+}4,\ 1{+}3\,] = [5,\ 4]$

<div class="insight">

The single input $[1,2,3,4]$ now plays **three roles** — query $[4,5]$, key $[6,5]$, value $[5,4]$ — like one actor in three costumes. The full $Q,K,V$ are this same calculation stacked over every token.

</div>

---

# One attention head, in PyTorch

```python
class AttentionHead(nn.Module):
    def __init__(self, d_in, d_k):
        super().__init__()
        self.Wq = nn.Linear(d_in, d_k, bias=False)   # the three costumes
        self.Wk = nn.Linear(d_in, d_k, bias=False)
        self.Wv = nn.Linear(d_in, d_k, bias=False)

    def forward(self, x):                            # x: (n, d_in)
        Q, K, V = self.Wq(x), self.Wk(x), self.Wv(x)
        scores  = Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))
        weights = scores.softmax(dim=-1)             # (n, n), rows sum to 1
        return weights @ V                           # (n, d_k)
```

Three `nn.Linear` layers and two matrix multiplies — that is the **entire** mechanism. The rest of this lecture explains the two lines inside `forward`.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1 · a tiny 3-token self-attention.** After projection, a 3-token sequence has

$$Q=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad K=\begin{pmatrix}1&0\\0&1\\1&1\end{pmatrix},\quad V=\begin{pmatrix}1&2\\3&4\\5&6\end{pmatrix},\quad d_k=2.$$

Compute the **output for token 1** (row 1): (a) its scores $q_1\!\cdot\!k_j$, (b) scale by $\sqrt{d_k}$ and take the softmax, (c) form the weighted sum of value rows.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**(a) Scores.** $q_1=[1,0]$, so $q_1\!\cdot\!k_j = [\,1,\ 0,\ 1\,]$ against $k_1,k_2,k_3$.

**(b) Scale & softmax.** Divide by $\sqrt2\approx1.414$: $[0.707,\ 0,\ 0.707]$.
$\exp \approx [2.03,\ 1.00,\ 2.03]$, sum $\approx 5.06$ $\Rightarrow$ weights $\approx [0.40,\ 0.20,\ 0.40]$.

**(c) Weighted sum of values.**
$$0.40\,[1,2] + 0.20\,[3,4] + 0.40\,[5,6] = [\,0.40{+}0.59{+}2.01,\ \ 0.80{+}0.79{+}2.41\,] \approx [\mathbf{3.0},\ \mathbf{4.0}].$$

<div class="keypoint">

Token 1 reads mostly $v_1$ and $v_3$ (equal weight), and their average is exactly $v_2=[3,4]$. The output is a **contextualized blend** — scores → softmax → weighted values, the same three steps for every row.

</div>

---

# Worked example · the *whole* attention matrix

Practice problem 1 read out **row 1**. Turn the same crank for rows 2 and 3 and you have the complete self-attention layer — every output token in one $3\times3$ pass.

<div class="math-box">

Scaled scores $QK^\top/\sqrt2$, then row-softmax (each row sums to 1):
$$A=\begin{pmatrix}0.40&0.20&0.40\\[2pt]0.20&0.40&0.40\\[2pt]0.25&0.25&0.50\end{pmatrix}$$

Weighted values $AV$ with $v_1=[1,2],\ v_2=[3,4],\ v_3=[5,6]$:
$$AV \approx \begin{pmatrix}3.0&4.0\\[2pt]3.4&4.4\\[2pt]3.5&4.5\end{pmatrix}$$

</div>

<div class="insight">

Token 3's query matches key 3 best, so it pulls hardest on $v_3$ and drifts toward $[5,6]$. **Every row is the same four steps — scores → scale → softmax → blend — run in parallel, no loop over tokens.** This $3\times3$ matrix *is* the whole self-attention layer for a 3-word sentence.

</div>

---

<!-- _class: section-divider -->

## Part 3 · The scaled dot-product, and why $\sqrt{d_k}$

---

# Scaled dot-product attention, step by step

![w:900px](figures/lec12/svg/attention_scaled_dotproduct.svg)

<div class="math-box">

$$\text{Attn}(Q,K,V) \;=\; \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

</div>

$QK^\top$ scores every query against every key ($n\times n$) · divide by $\sqrt{d_k}$ · row-softmax → weights · multiply by $V$ → the weighted values. Four operations, one line.

---

# Worked numeric · the four steps

Tiny example · 2 tokens, $d_k=2$, $\sqrt{d_k}=\sqrt2$.
$Q=\begin{pmatrix}1&0\\0&1\end{pmatrix}$, $K=\begin{pmatrix}1&1\\2&0\end{pmatrix}$, $V=\begin{pmatrix}0.1&0.2\\0.3&0.4\end{pmatrix}$.

**1 · Scores.** $\;QK^\top=\begin{pmatrix}1&2\\1&0\end{pmatrix}$ — token 1's query prefers token 2 (2 > 1).

**2 · Scale.** $\;QK^\top/\sqrt2=\begin{pmatrix}0.707&1.414\\0.707&0\end{pmatrix}$.

**3 · Row-softmax.** Row 1 $\to[0.33,0.67]$, Row 2 $\to[0.67,0.33]$.

**4 · Weighted values.** $AV=\begin{pmatrix}0.234&0.334\\0.166&0.266\end{pmatrix}$ — token 1's output is $33\%\,v_1 + 67\%\,v_2$.

---

# Why softmax can turn "spiky"

<div class="insight">

**Analogy · grading on a curve.**
- Scores in $1$–$10$: a 7 vs an 8 are close. The curve stays *smooth* — everyone gets some credit.
- Scores in $1$–$1000$: 700 vs 800 are worlds apart. The 801 takes everything; everyone else gets $\approx 0\%$. The curve goes **spiky**.

</div>

Unscaled dot products behave like the **second** case as $d_k$ grows: the scores spread out, softmax collapses toward one-hot, and gradients through it die. The $\sqrt{d_k}$ pulls us back to the first case.

---

# Why $\sqrt{d_k}$: the variance argument, kept intuitive

![w:900px](figures/lec12/svg/sqrt_dk_scaling.svg)

A score $q\cdot k=\sum_{i=1}^{d_k} q_i k_i$ sums $d_k$ independent products. Sum of $d_k$ zero-mean unit-variance terms has variance $d_k$, so the **spread grows like $\sqrt{d_k}$**. Dividing by $\sqrt{d_k}$ restores variance $\approx 1$ — and it is **dimension-invariant**: one block works at $d_k=64$ or $4096$ with no temperature retuning.

---

# Worked numeric · the scaling factor in action

$d_k=256 \Rightarrow \sqrt{d_k}=16$. A typical raw-score row for one query: $[15.5,\ -16.1,\ 2.3]$.

<div class="columns">
<div>

**Without scaling.**
$\exp(15.5)\approx 5.4\times10^6$ dominates.
Softmax $\approx [0.999998,\ 0,\ 0.000002]$ — essentially **one-hot** → gradient $\approx 0$ → learning stalls.

</div>
<div>

**With scaling** (divide by 16).
$[0.97,\ -1.01,\ 0.14]$, $\exp\approx[2.64,0.36,1.15]$.
Softmax $\approx [0.64,\ 0.09,\ 0.27]$ — **soft**, and gradients flow.

</div>
</div>

<div class="keypoint">

Same logits, one division: saturated → healthy. The $\sqrt{d_k}$ is a *temperature derived from variance*, not a hyperparameter you tune.

</div>

---

# Intuition · $\sqrt{d_k}$ is just a *temperature*

Softmax has a temperature knob $T$: $\ \text{softmax}(z/T)$. Large $T$ → smooth, near-uniform weights; small $T$ → peaky, near one-hot. Scaling the scores by $1/\sqrt{d_k}$ **is** setting $T=\sqrt{d_k}$.

<div class="insight">

The twist: this temperature is **not** a knob you sweep. The variance argument *derives* $T=\sqrt{d_k}$ as the exact value that holds the score spread near $1$ for **every** width — so one setting works from a 2-token toy to a 4096-dim production head. A temperature you would normally tune here falls straight out of the math.

</div>

<div class="notebook">

**🎛 Interactive · softmax & temperature** — slide $T$ from peaky to uniform and watch the weight distribution (and its entropy) move; then set $T=\sqrt{d_k}$ and see why the scaled scores stay soft. Same knob drives LLM sampling and distillation. *(Interactive Lab · `interactive/softmax-temperature`)*

</div>

---

# Does $\sqrt{d_k}$ change *which* token wins?

A fair worry: if we divide every score by $\sqrt{d_k}$, do we change the model's *choice*? Take the saturated row $[15.5,\,-16.1,\,2.3]$ from two slides ago.

<div class="columns">
<div>

**Ranking is untouched.** Dividing every entry by the same $16$ is monotonic — token 1 still has the largest score, token 2 the smallest. The **argmax never moves.**

</div>
<div>

**Only the *sharpness* changes.** Unscaled → $[\approx1,\,0,\,\approx0]$ (one-hot). Scaled → $[0.64,\,0.09,\,0.27]$ (soft). Same winner — but now the runners-up still contribute, and gradients survive.

</div>
</div>

<div class="keypoint">

Scaling is a **temperature, not a re-vote**: it decides *how confidently* attention commits, never *what* it commits to. That is exactly why it can be derived from variance alone, without ever touching the model's learned preferences.

</div>

---

# Why one-hot is the failure we're avoiding

If softmax outputs go (nearly) one-hot, attention picks **one** position and ignores the rest. Two things break at once:

1. **Information loss** — every other value vector is discarded; the "weighted blend" degenerates into a hard pick.
2. **Dead gradients** — a saturated softmax has near-zero Jacobian, so no learning signal reaches $W_Q, W_K$.

<div class="insight">

Attention only *works* while it stays soft. This is why the search-engine intuition — *mix the top matches* — beats *return the single top page*: the mix is what keeps the operation differentiable and trainable.

</div>

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2 · why $\sqrt{d_k}$ keeps softmax from saturating.** Query and key entries are i.i.d. $\mathcal N(0,1)$ and $d_k=64$.

(a) What is the standard deviation of an **unscaled** score $q\cdot k$?
(b) Roughly what range do the scores span, and why does feeding those into softmax make it (nearly) one-hot?
(c) Show that dividing by $\sqrt{d_k}$ restores unit variance, and say why this fix needs **no** per-dimension retuning.

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

**(a)** $q\cdot k=\sum_{i=1}^{64} q_i k_i$. Each product has mean $0$, variance $1$; independent, so $\text{Var}=64$ and $\text{std}=\sqrt{64}=\mathbf{8}$.

**(b)** Scores routinely land at $\pm 2$–$3$ std, i.e. **$\pm 16$ to $\pm 24$**. Exponentiating gaps that large makes one term dwarf all others → softmax $\approx$ one-hot → saturated, near-zero gradient.

**(c)** $\text{Var}\!\left(\dfrac{q\cdot k}{\sqrt{d_k}}\right)=\dfrac{d_k}{d_k}=1$ for **any** $d_k$. Because the correction is exactly $\sqrt{d_k}$, the same attention block stays well-behaved at $d_k=64$ or $4096$ — dimension-invariant, no tuning.

<div class="keypoint">

The scale isn't cosmetic: it is what keeps the softmax in the regime where gradients survive and every value still contributes.

</div>

---

<!-- _class: section-divider -->

## Part 4 · Self-attention vs cross-attention

---

# Cross-attention: the decoder reads the encoder

Same machinery — the only question is *who asks* and *who answers*.

<div class="columns">
<div>

**Cross-attention** (seq2seq + attention, translation)
$$Q = \text{decoder state},\quad K,V = \text{encoder states}$$
Queries come from the current **target** position; keys/values come from **all source** positions.

</div>
<div>

This is exactly the first heatmap: **one distribution per target step over the source**. It replaces L13's single $\mathbf c$ — the decoder now reads a *fresh* weighted blend of source words at every output step.

</div>
</div>

<div class="insight">

Cross-attention is the literal implementation of "look back at the source." The bottleneck is gone: nothing is compressed, everything stays readable.

</div>

---

# Self-attention: the "bank" disambiguation

<div class="keypoint">

How do you know *"bank"* means a financial institution in *"the **bank** approved the **loan**"*? You look at the other words — *"loan"* fixes the meaning. In *"he sat on the river **bank**"*, *"river"* fixes the other meaning.

</div>

**Self-attention** is exactly this: every word looks at every *other* word **in the same sequence** to compute its context-specific meaning. The canonical *"the animal didn't cross the street because **it** was too tired"* needs it — attention resolves *"it"* by attending to *"animal"*.

---

# Self-attention: a sequence attends to itself

Now all three projections come from **the same** input sequence $X$:

<div class="math-box">

$$Q = X W_Q,\qquad K = X W_K,\qquad V = X W_V$$

$$QK^\top/\sqrt{d_k}=\begin{bmatrix} s_{11}&s_{12}&s_{13}\\ s_{21}&s_{22}&s_{23}\\ s_{31}&s_{32}&s_{33}\end{bmatrix}\quad\text{row }i:\ \text{how much token }i\text{ looks at each token}$$

</div>

Softmax each row → an $n\times n$ attention matrix; multiply by $V$ → each output row is a contextualized blend of value rows.

<div class="keypoint">

Self-attention lets every position aggregate information from every other position **in parallel, with no recurrence.** That parallelism is what let attention replace RNNs — and it is the engine L15 stacks into the Transformer.

</div>

---

# Self vs cross, in one line

Same three steps, same softmax, same weighted sum. The *only* difference is **where the queries and the keys/values come from.**

<div class="columns">
<div>

### Self-attention · *look within*
$Q,K,V$ all from **one** sequence.
"Let every word refine its own meaning using the other words *beside it*." Resolves *"it"*, disambiguates *"bank"*.

</div>
<div>

### Cross-attention · *look across*
$Q$ from sequence A, $K,V$ from sequence B.
"Let each target word pull the source words it needs." Translation, captioning, retrieval.

</div>
</div>

<div class="keypoint">

**Self = a sequence talking to itself; cross = one sequence reading another.** Swap where $Q$ and $K,V$ are sourced and the *identical* module does both jobs — the projections, the $\sqrt{d_k}$, the softmax never change.

</div>

---

# The kinds of attention you'll meet

| Type | $Q$ from | $K,V$ from | Lives in |
|------|:--------:|:----------:|:--------:|
| Encoder self-attention | encoder | encoder (all) | Transformer encoder, BERT |
| Decoder self-attention (**causal**) | decoder (past only) | decoder (past only) | GPT, Transformer decoder |
| Cross-attention | decoder | encoder (all) | translation, Whisper, T5 |

Two more masks you'll meet in real code: the **padding mask** (ignore `<pad>` tokens when batching unequal lengths) and **attention dropout** (randomly zero some weights while training). All of them are the *same* module — only the mask changes.

---

# Self-attention vs convolution: same goal, different bias

<div class="columns">
<div>

### Convolution
- each output depends on a **fixed local window**
- inductive bias: *locality*, baked in
- parameters: a shared kernel

</div>
<div>

### Self-attention
- each output can depend on **all** positions
- inductive bias: *learned* from data
- parameters: the $Q,K,V$ projections

</div>
</div>

<div class="insight">

Convolution hard-codes "nearby tokens matter"; self-attention lets the data decide whether near or far matters. With enough data the *learned* bias wins — but with little data the hand-built bias still wins (ViT needed ~300M images to beat ResNet). That trade-off is the whole 2017–2025 vision arc in one line.

</div>

---

# Self-attention is permutation-equivariant — so L15 must add positions

Look again at $\text{softmax}(QK^\top/\sqrt{d_k})\,V$: it is built entirely from **dot products and sums**, which don't care about order. Shuffle the input tokens and every output row shuffles with them — *identically*.

<div class="math-box">

$$\text{Attn}(PX) = P\,\text{Attn}(X)\quad\text{for any permutation }P.$$

</div>

<div class="keypoint">

Self-attention treats its input as a **set**, not a sequence. That is a *feature* for set-structured data — but a **bug** for language, where *"dog bites man"* $\ne$ *"man bites dog"*. The model has no idea which token came first.

</div>

The fix isn't inside attention — it's to **stamp order into the embeddings** with positional encodings, the first thing L15 adds on top of this engine.

<div class="notebook">

**🎛 Interactive · positional encoding** — see the sinusoidal position signals that break the tie, and how they let a permutation-blind attention layer recover word order. *(Interactive Lab · `interactive/positional-encoding`)*

</div>

---

<!-- _class: section-divider -->

## Part 5 · The causal mask and the $O(n^2)$ wall

---

# Causal self-attention: no peeking at the future

![w:860px](figures/lec12/svg/causal_mask.svg)

<div class="insight">

When writing the next word of *"The quick brown fox jumps…"*, you can only use what you've **already** written. A **causal mask** covers the future with cardboard: for *"fox"* you may read *"The quick brown fox"* but everything after is hidden. Otherwise the model would cheat by reading the answer.

</div>

---

# How the $-\infty$ mask works, by hand

Raw scores (3 tokens, $d_k=1$ so no scaling):
$S=\begin{pmatrix}2.1&1.5&0.4\\0.8&3.1&1.2\\1.4&0.7&2.5\end{pmatrix}$

**Step 1 · mask the upper triangle** (set future positions to $-\infty$):
$S'=\begin{pmatrix}2.1&-\infty&-\infty\\0.8&3.1&-\infty\\1.4&0.7&2.5\end{pmatrix}$

**Step 2 · row-softmax.** Since $\exp(-\infty)=0$:
Row 1 $\to[1,0,0]$; Row 2 $\to[0.09,0.91,0]$; Row 3 $\to[0.22,0.11,0.67]$.

The $-\infty$ forces every future weight to **exactly zero** — token 1 sees itself, token 2 sees 1–2, token 3 sees all. Rows still sum to 1.

---

# The other mask you'll meet — and its NaN trap

Batching sentences of unequal length means padding short ones with `<pad>` tokens. A **padding mask** sets those columns to $-\infty$ so real tokens never attend to filler — the same $-\infty$ trick, a different pattern (whole columns, not a triangle).

<div class="warning">

**Watch the fully-masked row.** If *every* key for some query is masked, that row is all $-\infty$; softmax over all $-\infty$ is $0/0 =$ **NaN**, and the NaN spreads through the whole batch. It bites when a query position is itself padding, or when causal + padding masks overlap on row 0. *Fixes:* never let a query attend to nothing (keep the diagonal open), or clamp with a large finite negative instead of true $-\infty$.

</div>

Causal, padding, local, cross — all the same module. **Only the mask changes.**

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3 · the causal mask for length 4.** A GPT-style decoder processes a length-4 sequence with self-attention.

(a) Write the $4\times4$ mask $M$ that additive-masks the scores (entries $0$ where attention is allowed, $-\infty$ where it is forbidden).
(b) How many entries are masked, and what does the specific entry $M_{2,4}$ prevent?
(c) In one line: what would go wrong at *training* time if you dropped the mask entirely?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

**(a)** Allow $j\le i$, forbid the strict upper triangle $j>i$:
$$M=\begin{pmatrix}0&-\infty&-\infty&-\infty\\0&0&-\infty&-\infty\\0&0&0&-\infty\\0&0&0&0\end{pmatrix}$$

**(b)** The strict upper triangle has $\binom{4}{2}=\mathbf{6}$ masked entries. $M_{2,4}=-\infty$ prevents **token 2 from attending to token 4** — reading a word three positions in its future.

**(c)** With no mask, every position sees the whole sequence, so predicting token $t$ can just *copy* token $t{+}1$ from the input. Training loss collapses to near-zero, and the model learns **nothing** about generation — it fails the instant it must predict without the answer present.

<div class="keypoint">

Same module, one triangular mask: BERT-style (bidirectional, no mask) vs GPT-style (causal). The mask is the entire difference.

</div>

---

# Practice problem 4

<div class="popquiz">

**Practice problem 4 · cross-attention shapes.** A translation decoder attends to its encoder. The **source** has $m=5$ tokens; the **target** is at $n=3$ positions. Model width $d_\text{model}=512$, projected to $d_k=d_v=64$.

Give the shape of: **(a)** $Q$, $K$, $V$; **(b)** the score matrix $QK^\top$, and the axis softmax runs over; **(c)** the attention output. **(d)** One line: which of these shapes would *change* if you switched to encoder **self**-attention?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 4

**(a)** $Q$ from the 3 target states $\to \mathbf{3\times64}$. $K,V$ from the 5 source states $\to \mathbf{5\times64}$ each.

**(b)** $QK^\top \to \mathbf{3\times5}$ — one row per *target* position, one column per *source* token. **Softmax runs along the 5 source columns**, so each target position gets its own distribution over the source.

**(c)** $AV \to \mathbf{3\times64}$: one contextualized vector per target position (a per-head $W_O$ then maps it back to $512$).

**(d)** Encoder self-attention sets source $=$ target, so $Q,K,V$ are all $5\times64$ and the score matrix becomes a **square $5\times5$** — the rectangle only appears when the two sequences differ.

<div class="keypoint">

Cross-attention scores are **rectangular** ($n_\text{target}\times n_\text{source}$); self-attention scores are **square** ($n\times n$). Same module, same softmax — only *who supplies $Q$ versus $K,V$* decides the shape.

</div>

---

# Intuition · every token compares to every token

Self-attention's cost comes straight from its superpower. To let position $i$ read position $j$ **for every pair**, you must actually *form* every pair.

<div class="math-box">

$n$ tokens → an $n\times n$ grid of scores → $n^2$ comparisons.
$$n=10\to100,\qquad n=1{,}000\to10^6,\qquad n=100{,}000\to10^{10}.$$

</div>

<div class="insight">

Double the context and you **quadruple** the work — every new token compares against *all* the old ones *and* itself. Convolution never pays this: it only compares within a fixed window. **Global reach is exactly what costs $O(n^2)$** — total connectivity vs quadratic cost is the tension the next slide, and all of L23, wrestles with.

</div>

---

# The $O(n^2)$ wall

Self-attention on a length-$n$ sequence builds the full $n\times n$ score matrix $QK^\top$:

- **memory** $O(n^2)$ · **compute** $O(n^2 d)$ · double the context → **4×** the cost.

<div class="warning">

At $n=8{,}192$, one head's score matrix has $8192^2\approx67$M entries — **~134 MB in fp16, per head, per layer**. At $n=1$M tokens it becomes a *trillion*-entry (terabyte-scale) matrix that naive attention simply cannot store. This wall motivates **FlashAttention** (tiled recompute), **sparse / linear attention**, and **KV-caching** — all in L23.

</div>

---

# Practice problem 5

<div class="popquiz">

**Practice problem 5 · the cost of more context.** A model runs self-attention with context length $n=1{,}024$, head dimension $d=64$, in **fp16** (2 bytes/entry).

**(a)** How many entries are in one head's score matrix $QK^\top$, and how much memory is that?
**(b)** You extend the context to $n=4{,}096$. By what factor does the score-matrix memory grow?
**(c)** In one line: why is the *compute* $O(n^2 d)$ and not $O(n^3)$?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 5

**(a)** The score matrix is $n\times n$ regardless of $d$: $\;1024^2=\mathbf{1{,}048{,}576}$ entries $\approx1.05$M. At 2 bytes each, $\approx\mathbf{2.1\ \text{MB}}$ — per head, per layer.

**(b)** Memory scales as $n^2$: $\left(\tfrac{4096}{1024}\right)^2 = 4^2 = \mathbf{16\times}$ (to $\approx33.6$ MB). The "double $\to 4\times$" rule, applied twice.

**(c)** Forming $QK^\top$ is $n^2$ dot products of length $d$ $\to O(n^2 d)$; the $d$ is the *work per pair*, not a third loop over tokens. Multiplying by $V$ costs another $O(n^2 d)$.

<div class="keypoint">

Notice the memory ($n^2$) is **independent of $d$** — it is the $n^2$ term, not the width, that explodes with context. Stack dozens of heads and layers and this is the wall FlashAttention, sparsity, and KV-caching exist to break (L23).

</div>

---

# Escaping the $n^2$ wall — a first look

Three families of fixes, all deferred to **L23** — but the intuition is already yours:

<div class="columns">
<div>

- **FlashAttention** — never store the full $n\times n$ matrix; tile the compute and recompute on the fly. *Same math, $O(n)$ memory.*
- **Sparse / local attention** — let each token attend to a *window* plus a few global tokens, not all $n$.

</div>
<div>

- **Linear attention** — reorder the products to skip the $n\times n$ matrix entirely, $O(n)$ compute.
- **KV-caching** — at generation time, reuse past keys/values instead of recomputing them every step.

</div>
</div>

<div class="notebook">

**🎛 Interactive · KV-cache** — step a decoder forward token by token and watch the key/value cache grow, turning an $O(n^2)$ regenerate into an $O(n)$ append — the trick behind fast LLM inference. *(Interactive Lab · `interactive/kv-cache`)*

</div>

<div class="notebook">

**📓 Notebook · attention by hand (masked)** — the same three-`nn.Linear` head, now with a causal mask: verify row $i$ attends only to positions $\le i$, and that removing the mask lets the model "cheat" on next-token prediction. *(ES 667 · `notebooks/12-attention-by-hand.ipynb`)*

</div>

---

# What you built — and what's next

One bottleneck, climbed one rung at a time:

**weighted sum of values → Q/K/V projections → scaled dot-product → self-attention → causal mask.**

<div class="keypoint">

Give each self-attention layer a few **heads**, stack a handful of layers, wrap each in **residual + LayerNorm**, and add **positional encodings** (self-attention is permutation-equivariant — it needs to be *told* the order) — and that is a **Transformer**. Attention is the engine; **L15 bolts on the chassis.**

</div>

**Next (L15):** the same machinery, assembled into the architecture behind BERT, GPT, and Claude — built from these parts, not re-derived.

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Attention is a soft, differentiable dictionary lookup: score a query against every key, softmax into weights, and read out a weighted sum of values.**

</div>

- **The whole idea** — output $=\sum_i \text{softmax}_i(q\cdot k_i)\,v_i$; the bottleneck of L13 disappears because every output can read every input directly.
- **Q, K, V** — three *learned* projections of the same input; keys for matching, values for content.
- **$\sqrt{d_k}$** — scores spread like $\sqrt{d_k}$; divide to keep softmax soft and gradients alive.
- **Self vs cross** — same sequence for $Q,K,V$ (self) vs decoder-queries-reading-encoder (cross).
- **Causal mask** — set the future to $-\infty$ so a decoder never peeks; the cost is the $O(n^2)$ wall.

**Next (L15):** stack it, multi-head it, norm it, add positions — the Transformer.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's own courses (IIT Gandhinagar) and standard references:

- **Attention mechanics, QKV, scaled dot-product, self/cross, causal mask, $O(n^2)$** — *ES 667 Attention materials*, N. Batra.
- **Pedagogical framing & sequencing** — A. Ng, *Deep Learning Specialization*, Course 5 (Sequence Models), Weeks 3–4.
- **The QKV visual walk-through & library/dictionary intuition** — J. Alammar, *The Illustrated Transformer*.
- **Attention as soft alignment (additive), the first heatmaps** — Bahdanau, Cho & Bengio, *"Neural Machine Translation by Jointly Learning to Align and Translate"* (2015).
- **Scaled dot-product attention, self-attention, the $\sqrt{d_k}$** — Vaswani et al., *"Attention Is All You Need"* (2017).

Interactive Lab · `interactive/attention`, `interactive/seq2seq-bottleneck`, `interactive/softmax-temperature`, `interactive/positional-encoding`, `interactive/kv-cache`. Notebook · `notebooks/12-attention-by-hand.ipynb`. Figures adapted from the ES 667 figure library. All source courses © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · variance of an unscaled dot product

Let $q,k\in\mathbb R^{d_k}$ have i.i.d. entries with $\mathbb E[q_i]=\mathbb E[k_i]=0$ and $\text{Var}(q_i)=\text{Var}(k_i)=1$, independent of each other. The score is $S=\sum_{i=1}^{d_k} q_i k_i$.

Each term: $\;\mathbb E[q_i k_i]=\mathbb E[q_i]\,\mathbb E[k_i]=0$, and since the pair is independent and zero-mean,
$$\text{Var}(q_i k_i)=\mathbb E[q_i^2 k_i^2]-0=\mathbb E[q_i^2]\,\mathbb E[k_i^2]=1\cdot1=1.$$

The $d_k$ terms are independent, so variances add:
$$\text{Var}(S)=\sum_{i=1}^{d_k}\text{Var}(q_i k_i)=d_k \quad\Longrightarrow\quad \text{std}(S)=\sqrt{d_k}.$$

At $d_k=512$ raw scores reach $\sim\pm22$, saturating softmax to one-hot. Dividing by $\sqrt{d_k}$ gives $\text{Var}(S/\sqrt{d_k})=1$ **for all $d_k$** — the scaling is derived, not tuned. *(Full treatment: Vaswani et al. 2017, §3.2.1.)*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · additive (Bahdanau) attention, the origin

Before the dot product won, Bahdanau et al. (2015) scored compatibility with a tiny learned MLP — a *learned* similarity that works even when query and key aren't in aligned spaces:

$$e_{t,i}=\mathbf v^\top\tanh(W_1 h_i + W_2 s_t),\qquad \alpha_{t,i}=\text{softmax}_i(e_{t,i}),\qquad c_t=\sum_i \alpha_{t,i}\,h_i.$$

Luong (2015) then showed a plain $e_{t,i}=s_t^\top h_i$ is as expressive once you have learned $W_Q,W_K$ projections — and it is one matrix multiply GPUs love, versus Bahdanau's $\approx 2d^2$ extra parameters plus an element-wise $\tanh$. Vaswani et al. (2017) kept the multiplicative form and added the single crucial ingredient — the $/\sqrt{d_k}$.

<div class="insight">

The arc in one line: **learned-MLP similarity → dot-product similarity → *scaled* dot-product.** The dot product survived because it is cheap and, with learned projections around it, loses nothing in expressiveness. *(Bahdanau et al. 2015; Luong et al. 2015; Vaswani et al. 2017.)*

</div>
