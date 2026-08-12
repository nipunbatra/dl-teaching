# Public Lecture 12 · Next-Token Prediction · Teaching Guide

*File family: L11. First-time-instructor companion for an 80-minute undergraduate deep-learning class.*

## Release tuple

- Deck source: `lecture11/L11-next-token.typ`
- Source SHA-256: `c42a4fb5ef49ddf8693973b61598893420841ec6f4c05b49c807bec0ebfe9efc`
- Notebook: `notebooks/L11/01_char_mlp_language_model.ipynb`
- Notebook SHA-256: `0bafad82d1055c4d23622491746a832228b234f35407e4cf2c143fc318b875f5`
- Public identity: Lecture 12; the physical L11 name is retained for repository compatibility.

## The spine in two sentences

A language model maps an observed prefix to a probability distribution over the next token. In one six-token character model, students will construct causal windows, look up and concatenate embeddings, compute an MLP forward pass and softmax, backpropagate `p-y` to the two rows that were read, then separate training, decoding, and evaluation before exposing the fixed-window model's blind spot.

## What changes from public Lecture 11

Lecture 11 placed one categorical classifier at each spatial location. Lecture 12 keeps logits, softmax, cross-entropy, and backpropagation, but changes the index set from space to ordered time and feeds predictions back during generation.

Say this explicitly:

> The local classification machinery is familiar. Order, causality, boundary tokens, and the autoregressive feedback loop are new.

## Student contract

By the end, students should be able to:

1. factor a sequence probability into next-token conditional probabilities;
2. turn a character sequence into causal context-target windows without target leakage;
3. explain why token ids are categorical addresses rather than numeric features;
4. trace `id → embedding rows → ordered context → MLP → logits → stable softmax`;
5. reproduce the complete `.n → a` forward and backward calculation;
6. state which embedding rows a lookup example updates and why;
7. compute token NLL, sequence probability, mean NLL, and perplexity under a declared protocol;
8. distinguish TRAIN, INFER, and EVAL uses of the same probability vector;
9. explain greedy choice, categorical sampling, and temperature without calling one universally safer;
10. diagnose a fixed-context model by holding its visible suffix fixed while changing the hidden prefix.

## Prerequisites to retrieve, not reteach

- categorical cross-entropy and stable softmax;
- matrix shapes and affine layers;
- ReLU forward/backward behavior;
- computation-graph backpropagation;
- gradient descent and Adam as update rules;
- train/validation separation and a sealed evaluation contract.

## Stable semantic grammar

Use the same roles as the deck:

| Role | Meaning |
|---|---|
| teal | observed context, true data, TRAIN |
| blue | model scores, probabilities, INFER |
| orange | target token or chosen token |
| green | correct or retained evidence |
| red | loss, error, or failed assumption |
| ink | EVAL and protocol conventions |

Keep three clocks visible in discussion:

- **TRAIN:** true context → distribution; compare with target; update weights.
- **INFER:** current context → distribution; choose a token; append it.
- **EVAL:** held-out true context → token NLL; aggregate; exponentiate to PPL.

## The held corpus and conventions

Vocabulary order is fixed throughout:

`[., a, e, i, n, v]` with ids `[0,1,2,3,4,5]`.

Training names:

`naveen, navin, neev, neena, vani, naina, avni, anvi, veena, navi`

Held-out names:

`navina, naveena, veenavi, anvina`

Toy boundary convention:

- `.` is a shared boundary symbol, not punctuation;
- repeat it twice as left padding because context length is `k=2`;
- use one `.` target to terminate a name;
- production systems normally separate BOS, EOS, PAD, and UNK.

Split whole names before creating windows. Never randomly split overlapping windows from the same name.

## Exact 80-minute route

The protected close is core. Totals are **CORE 55 + SHOULD 15 + OPTIONAL 10 = 80 minutes**.

| Clock | Priority | Teaching move |
|---|---:|---|
| 0–4 | Core | Bridge from spatial classification; hold the `.n` commitment |
| 4–8 | Core | Learning contract and TRAIN/INFER/EVAL clocks |
| 8–13 | Core | Corpus, tokenization, vocabulary, boundary convention |
| 13–19 | Core | Causal windows, chain rule, and the seven-row sequence ledger |
| 19–25 | Core | Token ids, one-hot vectors, embedding lookup |
| 25–31 | Core | Concatenation, tensor shapes, and 40-parameter count |
| 31–41 | Core | Exact `.n → a` forward pass, stable softmax, token CE |
| 41–48 | Core | Exact `p-y` backward pass and embedding-row update |
| 48–53 | Core | Sequence NLL/PPL, full-batch training evidence, evaluation contract |
| 53–58 | Should | Autoregressive loop, greedy versus categorical sampling |
| 58–63 | Should | Temperature as concentration, not correctness |
| 63–68 | Should | Suffix-blind diagnostic and opening revisit |
| 68–72 | Optional | Full output-gradient matrix |
| 72–76 | Optional | Follow-along notebook prediction protocol |
| 76–78 | Optional | Bengio provenance and extensions |
| 78–80 | Core | Fixed window → shared recurrent state; public Lecture 13 handoff |

## Opening commitment

Show context `.n` and vocabulary `[.,a,e,i,n,v]`. Before revealing weights, ask students to commit to three questions:

1. What six numbers must the model return?
2. Must greedy decoding and sampling choose the same token?
3. Which embedding rows can this example update?

Do not answer yet. Revisit all three near the end.

## Causal windows and sequence factorization

For `naveen`, the seven training pairs are:

| context | `..` | `.n` | `na` | `av` | `ve` | `ee` | `en` |
|---|---:|---:|---:|---:|---:|---:|---:|
| target | `n` | `a` | `v` | `e` | `e` | `n` | `.` |

The exact chain rule is

\[
p(x_{1:7}\mid B)=\prod_{t=1}^{7}p(x_t\mid B,x_{<t}).
\]

It assumes neither independence nor fixed length. The approximation begins only when the MLP discards all but the last two tokens:

\[
p_\theta(x_t\mid B,x_{<t})\approx p_\theta(x_t\mid x_{t-2:t-1}).
\]

## Why embeddings, and why concatenate

Token ids are lookup addresses. The fact that `v=5` and `n=4` does not mean their semantic distance is one.

The exact embedding table used for the hand calculation is:

| token | feature 1 | feature 2 |
|---|---:|---:|
| `.` | 1 | 0 |
| `a` | 0 | 1 |
| `e` | 1 | 1 |
| `i` | −1 | 1 |
| `n` | 0 | 2 |
| `v` | 2 | 0 |

For `.n`, read `e_.=(1,0)` and `e_n=(0,2)`. Concatenate rather than average:

\[
c=[e_.;e_n]=(1,0,0,2)^\top.
\]

Concatenation preserves which token occupied which slot. Averaging would erase that order.

## Shape and parameter ledger

With vocabulary `V=6`, embedding width `d=2`, context length `k=2`, and hidden width `H=2`:

| block | shape | count |
|---|---:|---:|
| embedding `E` | `6×2` | 12 |
| hidden `W_h,b_h` | `2×4`, `2` | 10 |
| output `W_o,b_o` | `6×2`, `6` | 18 |

Total: **40 parameters**.

## The exact `.n → a` forward pass

The hidden layer is

\[
W_h=\begin{bmatrix}.5&0&0&.25\\0&1&1&0\end{bmatrix},\qquad
b_h=(0,-1)^\top.
\]

Therefore

\[
u=W_hc+b_h=(1,-1)^\top,\qquad h=\operatorname{ReLU}(u)=(1,0)^\top.
\]

Output rows follow `[.,a,e,i,n,v]`:

\[
W_o=\begin{bmatrix}-1&0\\2&1\\0&-1\\-1&1\\0&0\\1&-1\end{bmatrix},
\qquad b_o=0.
\]

Thus

\[
z=(-1,2,0,-1,0,1)^\top.
\]

Use stable softmax by subtracting `max(z)=2`:

\[
p=(.028644,.575333,.077863,.028644,.077863,.211653).
\]

For target `a`,

\[
\mathcal L=-\log p_a=.552806\text{ nats}.
\]

Pause on the distinction: logits are unconstrained scores; probabilities are normalized and sum to one.

## The exact backward pass

For one-hot target `a`,

\[
\nabla_z\mathcal L=p-y
=(.028644,-.424667,.077863,.028644,.077863,.211653).
\]

Continue:

\[
\nabla_h\mathcal L=W_o^\top(p-y)=(-.694969,-.685539)^\top.
\]

The ReLU mask from `u=(1,-1)` is `(1,0)`, so

\[
\nabla_u\mathcal L=(-.694969,0)^\top.
\]

Then

\[
\nabla_c\mathcal L=W_h^\top\nabla_u\mathcal L
=(-.347485,0,0,-.173742)^\top.
\]

Split and scatter only into rows that were read:

\[
\nabla_{e_.}\mathcal L=(-.347485,0),\qquad
\nabla_{e_n}\mathcal L=(0,-.173742).
\]

With plain gradient descent and `η=.1`:

\[
e_.\leftarrow(1.03475,0),\qquad e_n\leftarrow(0,2.01737).
\]

The target token's output row receives a gradient, but embedding row `a` does not: its embedding was not read in this example.

## One sequence ledger through probability and PPL

Use correct-target probabilities

`(.30, .575333, .70, .60, .50, .40, .80)`.

Only `.575333` comes from the exact weights; the other six are declared illustrative values.

\[
p(\texttt{naveen.}\mid\texttt{..})=.0115987,
\]

\[
\text{total NLL}=4.45686,
\quad \text{mean NLL}=.636694,
\quad \text{PPL}=e^{.636694}=1.89022.
\]

This is one illustrative name, not the trained notebook's validation score.

## Training and evaluation evidence

The notebook widens the same architecture to `d=4,H=24` and runs **300 full-batch Adam updates** over all 56 training pairs.

At the same final post-update parameter snapshot:

| split | NLL | PPL |
|---|---:|---:|
| train | .645308 | 1.90657 |
| four held-out names | .677965 | 1.96986 |

The notebook also prints `.645279/1.90652`: that is the pre-update objective used to compute Adam update 300, not the final post-update snapshot. Do not pair it with the final validation result.

Perplexity comparisons require the same tokenizer, boundary policy, held-out corpus, set of non-padding targets, and token-weighted reduction. A uniform six-token model has NLL `log 6=1.79176` and PPL `6`.

## Autoregressive decoding

At inference:

1. read the current two-token context;
2. compute logits and probabilities;
3. choose or sample one token;
4. append it and drop the oldest context token;
5. stop at `.` or a maximum length.

Greedy decoding chooses `a` because it has the largest probability. Categorical sampling need not: with cumulative intervals and `u=.65`, the exact distribution selects `e`.

## Temperature, stated precisely

\[
p_i(\tau)=\operatorname{softmax}(z/\tau)_i,\qquad \tau>0.
\]

- `τ=.5` concentrates mass: `p(a)=.850`.
- `τ=1` keeps the learned distribution: `p(a)=.575`.
- `τ=2` flattens it: `p(a)=.359`.

For positive temperature, token ranking is unchanged. Lower temperature is not inherently safer or more correct; if the top-ranked token is wrong, it can make that error more consistent.

## The fixed-window diagnostic

Prefixes `na` and `veena` both expose the same final context `na`, so the trained model returns exactly

\[
p=(.418671,9.91\times10^{-7},3.77\times10^{-6},.142728,
3.17\times10^{-5},.438565).
\]

Use the recurring diagnostic:

| symptom | suspect | test |
|---|---|---|
| different long prefixes yield identical distributions | useful evidence is outside the fixed window | hold suffix fixed, change only earlier tokens, compare `p` exactly |

If `p` cannot change, more optimizer tuning cannot teach the architecture to see the hidden prefix.

## Notebook choreography

Use the one Colab in three passes:

1. **Before execution:** predict the seven windows, concatenated context, active ReLU unit, target-gradient sign, and embedding rows that move.
2. **Mechanics:** run the exact forward/backward cells and reconcile every array with the slide ledger.
3. **Evidence:** run full-batch training, compare same-snapshot train/validation metrics, generate with multiple temperatures, and test `na` versus `veena`.

Require an explain-back after each pass. The notebook is deterministic, downloads nothing, retains all outputs, and uses NumPy plus Matplotlib.

## Misconceptions to catch

- **“A larger token id is a larger input.”** No; ids address rows.
- **“The target embedding row must update.”** Only if that token was read in the context.
- **“Chain-rule factorization assumes token independence.”** No; each exact factor conditions on all previous tokens.
- **“Perplexity is the probability of the sequence.”** No; it exponentiates token-mean NLL.
- **“Greedy and sampling should agree.”** Only if the sampled uniform draw lands in the top token's interval.
- **“Lower temperature improves the model.”** It changes decoding concentration, not learned weights or ranking.
- **“More training can recover hidden-prefix information.”** Not when the architecture never receives it.
- **“The notebook uses minibatches.”** It uses all 56 training windows on every Adam step.

## If a student asks

**Why use a shared `.` for start and end?**

It keeps the hand calculation within six vocabulary items. Production systems normally separate BOS, EOS, PAD, and UNK.

**Why concatenate instead of average embeddings?**

Concatenation assigns each context position its own coordinate block; averaging erases order.

**Why subtract the maximum logit?**

Softmax is invariant to a shared shift, and the subtraction prevents exponential overflow.

**Why is the target coordinate of `p-y` negative?**

Gradient descent subtracts the gradient, so a negative target gradient raises the target logit.

**Does low validation PPL prove the names are good?**

No. Four held-out toy names provide only a narrow deterministic check.

**Why not keep increasing `k`?**

The hidden block grows with `Hkd`, positions still use different columns, and the maximum history remains fixed at construction.

## Short routes

### 55-minute route

Keep the opening commitment, causal windows, embedding/concatenation, full `.n → a` forward and backward pass, sequence NLL/PPL, the three clocks, one decoding example, the suffix-blind test, and the RNN handoff. Skip the full output-gradient matrix and most temperature arithmetic.

### 35-minute emergency route

Use the seven-row window table, show the already-computed `c,h,z,p`, derive only `p-y` and the embedding rows that move, state the NLL/PPL contract, contrast TRAIN with INFER, demonstrate suffix blindness, and hand off to recurrent state.

## Before-class checklist

- Open the exact Colab and verify all 12 code cells retain sequential counts with no warning output.
- Keep vocabulary order `[.,a,e,i,n,v]` visible while doing vector arithmetic.
- Decide whether students calculate the logits individually or in pairs.
- Do not compare perplexities across a different tokenizer or corpus.
- Keep the opening answer hidden until the revisit.

## Primary source

- Yoshua Bengio, Réjean Ducharme, Pascal Vincent, and Christian Jauvin, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html), JMLR 2003.

## Closing and public Lecture 13 handoff

Revisit the opening:

- output: six probabilities `(.029,.575,.078,.029,.078,.212)`;
- choice: greedy chooses `a`, while the declared sample chooses `e`;
- learning: lookup gradients update rows `.` and `n`.

Close with:

> The next-token objective stays. Next lecture replaces a fixed context window with one shared recurrent update that carries state across a variable-length stream.
