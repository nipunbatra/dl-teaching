Of course. Here is a concrete rewrite plan for the lecture on Tokenization and Pretraining Paradigms, following your specified format.

---

## SLIDE · "A concrete failure · character LMs"

**CURRENT PROBLEM**
The calculation comparing character-level vs. word-level attention cost (`36M` vs. `1M`) is presented without showing the work, making it seem like a magic number. Students need to see the simple math to trust the conclusion.

**REWRITE**
Train a character-level Transformer on Wikipedia. It *works* — GPT-like samples of readable English.

But it's expensive because attention complexity is $O(\text{sequence length}^2)$. Let's compare:

- **Character-level model:**
  - A 1000-word page is roughly 6000 characters (avg. 5 chars/word + space).
  - Sequence length $N=6000$.
  - Attention matrix entries: $N^2 = 6000 \times 6000 = \textbf{36,000,000}$ per layer.

- **Word-level model:**
  - The same 1000-word page is 1000 tokens.
  - Sequence length $N=1000$.
  - Attention matrix entries: $N^2 = 1000 \times 1000 = \textbf{1,000,000}$ per layer.

The character model is **36 times more expensive** in its attention layers.

Plus · character LMs have to *learn* that "t-h-e" is a meaningful unit. That's model capacity wasted re-learning the concept of words.

<div class="insight">

Subwords compromise: keep common sequences as one unit (saving sequence length), split rare sequences (keeping open vocabulary). The winning middle path.

</div>

**FIGURE**
A two-panel diagram. The left panel shows a long sequence of characters: `T-h-e- -q-u-i-c-k- ...`. Below it, a large $N \times N$ grid is shaded in, labeled "$6000 \times 6000 = 36M$ computations". The right panel shows a short sequence of word tokens: `The-quick-brown-...`. Below it, a much smaller $N \times N$ grid is shaded in, labeled "$1000 \times 1000 = 1M$ computations".

---

## SLIDE · "Worked BPE · "low lower newest widest""

**CURRENT PROBLEM**
This slide is the core of the BPE explanation, but it's a "tell, don't show." It presents the final merges without walking through the crucial intermediate step: counting the pairs to decide *which* merge to make. The logic is completely opaque to a new student.

**INSERT BEFORE**
**Intuition: Data Compression**
Think of BPE like a basic file compressor. If you see the string "ABCABCABC" a lot, you can save space by defining a new symbol `Z = "ABC"` and rewriting the file as "ZZZ". BPE does the same thing for language: it finds the most common sequence of characters (like `th`) and "compresses" it into a single new token.

**REWRITE**
Let's trace BPE step-by-step on the corpus: `"low lower newest widest"`.

**Step 0: Initialize Vocabulary and Corpus**
- Initial vocab is just the unique characters: `l, o, w, e, r, n, s, t, w, i, d`.
- Split each word into characters, adding a special `</w>` end-of-word marker.
  - `l o w </w>`
  - `l o w e r </w>`
  - `n e w e s t </w>`
  - `w i d e s t </w>`

**Step 1: Count Pairs and Merge**
- Count all adjacent pairs:
  - `(l, o): 2`
  - `(o, w): 2`
  - `(e, s): 2`
  - `(s, t): 2`
  - `(w, </w>): 1`
  - `(w, e): 1`
  - ... and so on for all other pairs.
- The most frequent pairs are `(l,o)`, `(o,w)`, `(e,s)`, and `(s,t)`. Let's break the tie and pick `(e,s)`.
- **Merge 1: `(e, s) -> "es"`**. Add "es" to vocab.
- Our corpus is now:
  - `l o w </w>`
  - `l o w e r </w>`
  - `n e w es t </w>`
  - `w i d es t </w>`

**Step 2: Recount and Merge Again**
- Re-count all adjacent pairs in the *new* corpus:
  - `(l, o): 2`
  - `(o, w): 2`
  - `(es, t): 2`  *(This is a new pair!)*
  - `(w, </w>): 1`
  - ... etc.
- The most frequent pairs are `(l,o)`, `(o,w)`, and `(es,t)`. Let's pick `(es, t)`.
- **Merge 2: `(es, t) -> "est"`**. Add "est" to vocab.
- Our corpus is now:
  - `l o w </w>`
  - `l o w e r </w>`
  - `n e w est </w>`
  - `w i d est </w>`

**Step 3 & 4: Continue Merging**
- **Merge 3: `(l, o) -> "lo"`.** Most frequent pair is now `(l,o)`.
  - `lo w </w>`, `lo w e r </w>`, `n e w est </w>`, `w i d est </w>`
- **Merge 4: `(lo, w) -> "low"`.**
  - `low </w>`, `low e r </w>`, `n e w est </w>`, `w i d est </w>`

After 4 merges, we've learned the reusable subwords "es", "est", "lo", and the word "low" as single tokens.

**INSERT AFTER**
**Worked Numeric Example: Corpus `"hug bug rug"`**

1.  **Initial state:**
    `h u g </w>`, `b u g </w>`, `r u g </w>`

2.  **Count pairs:**
    - `(u, g): 3`
    - `(g, </w>): 3`
    - `(h, u): 1`
    - `(b, u): 1`
    - `(r, u): 1`

3.  **Merge 1:** `(u, g) -> "ug"`. (It appeared 3 times).
    - New state: `h ug </w>`, `b ug </w>`, `r ug </w>`

4.  **Count pairs again:**
    - `(ug, </w>): 3`
    - `(h, ug): 1`
    - `(b, ug): 1`
    - `(r, ug): 1`

5.  **Merge 2:** `(ug, </w>) -> "ug</w>"`.
    - New state: `h ug</w>`, `b ug</w>`, `r ug</w>`

The algorithm has learned the reusable suffix `ug`.

**FIGURE**
A four-panel comic strip style diagram.
- **Panel 1:** Shows the initial list of character-split words. To the right is a table of pair counts, with `(e, s)` highlighted in red.
- **Panel 2:** The list of words now shows `e` and `s` merged into a single `es` token. The pair-count table to the right is updated, and now `(es, t)` is highlighted in red.
- **Panel 3:** Shows `est` merged. The count table is updated, `(l, o)` is highlighted.
- **Panel 4:** Shows `lo` merged. The count table is updated, `(lo, w)` is highlighted.
This visualizes the iterative process of "count, find max, merge, repeat."

---

## SLIDE · "BERT · encoder-only · masked LM"

**CURRENT PROBLEM**
The loss function $\mathcal{L}_\text{MLM} = -\sum_{i \in \text{masked}} \log P(x_i \mid x_{\setminus i})$ is abstract and assumes familiarity with log-likelihood losses and index notation. Students who just learned cross-entropy for logistic regression won't immediately connect this.

**INSERT BEFORE**
**Intuition: The Cloze Test**
Remember those "fill-in-the-blank" questions from school? "The quick brown ___ jumps over the lazy dog." To guess the blank, you use the words *before* it ("quick brown") and *after* it ("jumps over"). BERT does exactly this. We give it a sentence with a word hidden (`[MASK]`), and its only job is to guess the hidden word using all the surrounding context. The loss function just measures how good its guess was.

**REWRITE**
Let's build the math for the "fill-in-the-blanks" game.

1.  **Start with a sentence:**
    `x = ["The", "cat", "sat", "on", "the", "mat"]`

2.  **Mask a token:** We replace `x_4` ("on") with a special `[MASK]` token.
    `x' = ["The", "cat", "sat", "[MASK]", "the", "mat"]`

3.  **The Goal:** The model must predict the original token, `x_4`, given all the other tokens (the context). We want to maximize the probability of the correct word:
    $P(\text{token} = \text{"on"} \mid \text{context} = \text{"The cat sat ... the mat"})$

4.  **The Loss:** In ML, we minimize loss, not maximize probability. The standard way to do this is to use the **negative log probability** (which you'll remember from cross-entropy). For this one masked token, the loss is:
    $\text{loss}_4 = -\log P(x_4 \mid x_1, x_2, x_3, x_5, x_6)$
    Let's write the context as $x_{\setminus 4}$ (all $x$ *except* for index 4).
    $\text{loss}_4 = -\log P(x_4 \mid x_{\setminus 4})$

5.  **Total Loss:** We do this for ~15% of the tokens in the input. The total loss for the sentence is the sum of the losses for all the masked positions.
    $\mathcal{L}_\text{MLM} = \sum_{i \in \text{masked positions}} -\log P(x_i \mid x_{\setminus i})$

This objective forces the model to learn rich, bidirectional relationships between words.

**INSERT AFTER**
**Worked Numeric Example**
- Input: `...sat [MASK] the...` The correct word is "on".
- The Transformer outputs a vector of scores (logits) for the `[MASK]` position over the entire vocabulary.
  - `logits = {"on": 3.5, "above": 2.1, "under": 1.5, ...}`
- We convert these scores to probabilities using the `softmax` function:
  - `P("on") = exp(3.5) / (exp(3.5) + exp(2.1) + exp(1.5) + ...) ≈ 0.75`
  - `P("above") = exp(2.1) / (sum) ≈ 0.18`
  - `P("under") = exp(1.5) / (sum) ≈ 0.10`
- The loss for this prediction is the negative log probability of the *correct* word:
  - `Loss = -log(P("on")) = -log(0.75) ≈ 0.287`
If the model had been more confident (e.g., `P("on")=0.99`), the loss would be tiny (`-log(0.99) ≈ 0.01`).

**FIGURE**
A diagram showing a sentence "The cat sat [MASK] the mat." Arrows point from *every other word* ("The", "cat", "sat", "the", "mat") inward toward the `[MASK]` token. This visually emphasizes the "bidirectional context." A box pops up from `[MASK]` showing a probability distribution over a few vocabulary words (`on: 75%`, `above: 18%`, etc.), with the correct one highlighted.

---

## SLIDE · "GPT · decoder-only · causal LM"

**CURRENT PROBLEM**
Like the BERT slide, the CLM loss function $\mathcal{L}_\text{CLM} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1})$ is too dense. The summation over `t` and the conditional probability `P(x_t | x_{<t})` are not intuitive without a step-by-step buildup.

**INSERT BEFORE**
**Intuition: Smartphone Keyboard**
Think about the predictive text on your phone. When you type "I am heading to the", it suggests words like "gym", "store", or "movies". It's predicting the **next word** based only on the words you've *already typed*. It cannot see the future. This is exactly what a GPT model does, over and over again for every word in a sentence.

**REWRITE**
Let's build the math for the "predict-the-next-word" game. The model's task is to predict each token in a sentence, one at a time, using only the tokens that came before it.

1.  **Start with a sentence:**
    `x = ["The", "cat", "sat"]`

2.  **Break it down into a sequence of prediction problems:**
    - **Problem 1:** Predict the first word. Given nothing (`<start>`), what's the probability of "The"?
      - Loss 1: $-\log P(\text{"The"})$
    - **Problem 2:** Predict the second word. Given "The", what's the probability of "cat"?
      - Loss 2: $-\log P(\text{"cat"} \mid \text{"The"})$
    - **Problem 3:** Predict the third word. Given "The cat", what's the probability of "sat"?
      - Loss 3: $-\log P(\text{"sat"} \mid \text{"The", "cat"})$

3.  **Total Loss:** The model is trained on all these little prediction problems simultaneously. The total loss for the sentence is simply the sum of the individual losses at each step.
    $\mathcal{L} = (-\log P(x_1)) + (-\log P(x_2 \mid x_1)) + (-\log P(x_3 \mid x_1, x_2)) + \dots$

4.  **General Formula:** We can write this more compactly with a summation. For a sequence of length $T$:
    $\mathcal{L}_\text{CLM} = \sum_{t=1}^{T} -\log P(x_t \mid x_1, \ldots, x_{t-1})$
    The term $x_1, \ldots, x_{t-1}$ is just "all the tokens before position t," which we sometimes write as $x_{<t}$.

This objective forces the model to be an excellent generator of sequences.

**INSERT AFTER**
**Worked Numeric Example**
- Input context: `"The cat..."`. The next correct word is "sat".
- The model takes `"The cat"` as input and produces logits for the *next* token over the whole vocabulary.
  - `logits = {"sat": 4.0, "ran": 2.5, "jumped": 2.0, ...}`
- Convert to probabilities with `softmax`:
  - `P("sat" | "The cat") = exp(4.0) / (exp(4.0) + exp(2.5) + ...) ≈ 0.78`
  - `P("ran" | "The cat") = exp(2.5) / (sum) ≈ 0.16`
- The loss *for this specific step* is:
  - `Loss_at_t=3 = -log(P("sat" | "The cat")) = -log(0.78) ≈ 0.248`
- The total loss for the sentence is the sum of these losses computed at every position.

**FIGURE**
A diagram showing a sentence "The cat sat on the mat".
- At position 1 ("The"), there are no incoming arrows.
- At position 2 ("cat"), an arrow points from "The" to "cat".
- At position 3 ("sat"), arrows point from "The" and "cat" to "sat".
- This continues, with each token `t` only receiving arrows from tokens `1` to `t-1`. This visually represents the "causal" or "autoregressive" flow of information—the model can only look backward.

---

## SLIDE · "Compute economics · ballpark"

**CURRENT PROBLEM**
The formula `FLOPs ≈ 6 * N * D` is a crucial piece of LLM lore, but the `6` is a magic number. Deriving it from first principles (forward pass + backward pass cost) makes it understandable and memorable.

**INSERT BEFORE**
**Intuition: Rule-of-Thumb Estimation**
When you plan a road trip, you estimate the cost with a simple formula: `Total Cost ≈ (Distance / MPG) * Price_per_Gallon`. You don't count every single engine revolution. The formula for training cost (`6 * N * D`) is a similar rule of thumb for deep learning. It tells us the "compute cost" based on the model size (`N`) and the amount of data (`D`). Let's unpack where the `6` comes from.

**REWRITE**
Let's derive the rule of thumb for training FLOPs: `6 * N * D`.

- `N`: Number of parameters in the model (e.g., 70 billion).
- `D`: Number of tokens in the training dataset (e.g., 1.4 trillion).

**Where does the `6` come from?**
We need to calculate the floating-point operations (FLOPs) to process one token, then multiply by the total number of tokens.

1.  **Forward Pass Cost:** For a Transformer, most of the computation is in the two large matrix multiplications in the feed-forward network (FFN) layers. The other parts (attention, layer norms) are cheaper. A good approximation for the forward pass is **`2 * N` FLOPs per token**.

2.  **Backward Pass Cost:** The backward pass (for gradient calculation) involves matrix multiplications that are related to the forward pass operations. A standard rule of thumb is that the backward pass is about **twice as expensive** as the forward pass. So, the backward pass costs roughly **`4 * N` FLOPs per token**.

3.  **Total Cost per Token:**
    Cost per token = (Forward Pass Cost) + (Backward Pass Cost)
    Cost per token ≈ `2N + 4N = 6N` FLOPs.

4.  **Total Training Cost:**
    Total FLOPs = (Cost per token) × (Total tokens in dataset)
    **Total FLOPs ≈ `6 * N * D`**

This simple formula is surprisingly accurate for planning large-scale training runs.

**INSERT AFTER**
**Worked Numeric Example: Llama 2 7B**
Let's plug in the numbers for a smaller, well-known model.
- Model: Llama 2 7B
- Parameters `N` = 7 Billion = $7 \times 10^9$
- Dataset Tokens `D` = 2 Trillion = $2 \times 10^{12}$

**Calculation:**
`FLOPs ≈ 6 * N * D`
`FLOPs ≈ 6 * (7 * 10^9) * (2 * 10^{12})`
`FLOPs ≈ 84 * 10^{21}`
`FLOPs ≈ 8.4 * 10^{22}`

This number lets you estimate the hardware and time needed. For instance, if you have a cluster of GPUs that can deliver $10^{18}$ FLOPs per day, the training would take:
`Time ≈ (8.4 * 10^{22}) / (10^{18}) = 84,000` days for one GPU, or about 84 days for 1000 GPUs.

**FIGURE**
A simple visual breakdown.
- Title: "Why 6?"
- A box on the left labeled "Forward Pass (1 token)" with the text "Cost ≈ 2N FLOPs" inside. The "2N" comes from the two big matmuls in the FFN.
- A box on the right labeled "Backward Pass (1 token)" with the text "Cost ≈ 4N FLOPs (2x the forward pass)".
- An arrow from each box points to a final box at the bottom labeled "Total Cost per Token," containing the equation `2N + 4N = 6N`.
- A final arrow points from that box to the overall equation: "Total Training FLOPs ≈ `6 * N * D`".