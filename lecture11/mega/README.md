# L11M: From Characters to Transformers

This directory contains the three modular parts of the Typst mega-lecture. The
single entry point is:

```text
lecture11/L11M-from-characters-to-transformers.typ
```

The slide-by-slide source of truth is:

```text
lecture11/L11M-from-characters-to-transformers-outline.md
```

## Notation in Lectures 1 and 2

Keep lookup embeddings and attention values distinct. Before Q/K/V is
introduced, the averaging examples use `e_j`, not `v_j`. Keep `e_i` for the
token's embedding throughout the first attention calculation: we enrich it to
`e'_i = e_i + Δe_i`, rather than renaming it before explaining attention.

| Symbol | Meaning |
|---|---|
| `t_i`, `ids` | Token ID at position `i`; integer IDs run from `0` to vocabulary size minus one. |
| `E`, `e_i = E[t_i]` | Learned embedding table and its lookup row. A literal character subscript, such as `e_a`, names the token rather than its position. |
| `q_i`, `k_i`, `v_i` | Initially `e_i W_Q`, `e_i W_K`, `e_i W_V`: matching request, matching features, and information to mix. |
| `h_i` | Incoming weighted-value message for query position `i`, not a query-independent summary. |
| `Δe_i = h_i W_O` | Message mapped into embedding coordinates; `W_O` has shape `d_v × d`. |
| `e'_i = e_i + Δe_i` | Context-enriched embedding of this occurrence, read by the vocabulary predictor. This does not overwrite `E`. |
| `X` | Sequence matrix: `X = E[ids]`, with row `i` equal to `e_i` initially. It contains the current sequence, not the whole lookup table. |
| `Q`, `K`, `V`, `H` | Matrices stacking their corresponding lowercase rows in token order. |
| `ΔX`, `X'` | Matrices stacking `Δe_i` and `e'_i`. The matrix name `X` does not introduce a new single-token symbol. |
| `R`, `r_i` | Learned position table and its row; math position `i` uses Python index `i - 1`. Introduced later. |
| `e_i^(0) = e_i + r_i` | Position-aware input embedding; after positions are introduced, this is row `i` of `X` and supplies Q/K/V. |
| `e_i^(1) = e_i^(0) + Δe_i` | Embedding after the first contextual update with positions; this is then row `i` of `X'`. |

All vectors are rows: inputs multiply learned matrices on the right.
`X` is `T × d`; `Q,K` are `T × d_k`; `V,H` are `T × d_v`.
`A` is the `T × T` attention-weight matrix, and `H = A V`. `ΔX = H W_O`, `X' = X + ΔX`, and vocabulary logits are `Z = X' U + b`.
The optimizer updates `E`, `R` when used, and the learned weight/bias matrices;
it does not directly update the recomputed `Q,K,V,A,H,ΔX,X'`. Without positions,
the row calculation is `e_i → e'_i`; once positions are introduced, it is
`e_i^(0) → e_i^(1)`. These are forward-pass representations, not updates to
the stored embedding table.

Use `w` for the fixed context-window length, `d_h` for MLP hidden width, and
`|𝒱|` for vocabulary size. L1 uses `a_0` for the concatenated embedding row and
`a_1` for the hidden activation: `a_1 = σ(a_0 W_1 + b_1)`.
Reserve `v_i` for values. Before positions, setting `W_V = I` makes `v_i = e_i`.
After positions, the same identity map instead gives `v_i = e_i^(0)`.
In general, the value is a projection, not a renamed lookup embedding.

## Build

Run from the repository root:

```bash
typst compile --root . --input handout=true \
  lecture11/L11M-from-characters-to-transformers.typ \
  slides-pdf/L11M.pdf

typst compile --root . \
  lecture11/L11M-from-characters-to-transformers.typ \
  slides-pdf/L11M-presentation.pdf
```

## Audit

```bash
python3 scripts/audit_typst_slides.py \
  --max-raster-images 16 \
  slides-pdf/L11M.pdf \
  slides-pdf/L11M-presentation.pdf
```

The deck uses the shared course theme in `common/metropolis.typ`, including its
fonts, margins, dark-teal title bar, orange accent, and bottom slide number.
The step-by-step drawings and numerical examples follow the original
handwritten next-token notes.

Technical diagrams are native Typst/Fletcher vectors. The deck also includes
generated illustrations, so pass `--max-raster-images` with the expected image
count when running the audit (currently 8 in the handout and 12 in the
presentation, including transparency masks).

## One bank story: incoming information, prediction, and learning

The L2 core uses the bank example throughout. Two six-token prefixes introduce
context: `She deposited money in the bank` and `She watched water by the bank`.
They start with the same bank lookup row but offer different source information.
This contrast is qualitative; the untrained numerical model is not claimed to
disambiguate both meanings. Carry the finance prefix through every calculation.

Handout pages 60–81 build the intuition and distinguish the three roles.
Pages 82–88 calculate the unscaled six-token message. Pages 89–116 continue
that same example through a complete residual-attention language model.
Lecture 2 has 62 conceptual frames (handout pages 55–116); Lecture 3 starts
on page 117. The deck has 170 conceptual frames, 178 handout pages, and 436
progressive presentation pages.

Pages 71–75 slow down the query/key introduction before introducing values:
ask what could help after bank; calculate `q6=e6 W_Q` one coordinate at a time;
form source keys; compare money, the, and bank with that same query; then show
how the two prefixes offer different evidence to the same first-layer query.
The English topic/action/person questions are possible interpretations, not
literal decoded requests or claims about trained semantics. The selected toy
query gives scores `2,0,1` for those three sources, not probabilities; the
later softmax includes all six positions. Page 82 retains this query while
expanding to all six sources. Page 92 then generalizes the same query map to
She, money, and bank, instead of repeating bank's multiplication.

- `bank-worked.typ`: seven progressive hand calculations. Exponentials and
  normalization have separate frames. Input positions stay in the same order.
- `bank-training.typ`: 21 frames on embeddings, learned projections, the
  output map and residual addition, scaling, a fully specified vocabulary head,
  generation, loss, computational graph, autograd, and a verified SGD update.
- `bank-parallel.typ`: seven frames on shifted targets, causal masking,
  position information, numerical matrices, and all-position training.
- `bank-appendix.typ`: three optional frames on derivatives and dot-product
  intuition (pages 173–175). Manual gradient arithmetic stays out of the main
  lecture.
- `bank_training.py` and `bank-training-numbers.json`: executable calculations
  and full-precision evidence.

The vocabulary is `She, deposited, money, in, the, bank, ., and`. Input
embeddings, queries, keys, values, and messages have three coordinates in the
numerical example; vocabulary logits and probabilities have eight.
Initial `W_K=I`,
`W_Q=diag(1,0,0)`, and `W_V=diag(1,2,1)` reproduce the hand calculation.
`W_O=I` initially makes the incoming message itself the contextual update.
Every entry of these matrices is trainable; identity/diagonal forms are
teaching initializations, not architectural restrictions.

The warm-up explicitly rounds exponentials before adding them and uses rounded
attention weights. The trainable calculation uses `1/sqrt(3)` scaling and
float64 with no intermediate rounding:

- incoming message: `h6≈[1.017777,.571364,.285682]`;
- lookup embedding: `e6=[1,1,0]`;
- contextual embedding: `e6'≈[2.017777,1.571364,.285682]`;
- specified eight-column output head: `p(.)≈.410236`.

Generation holds parameters fixed. A clearly hypothetical sample `and` is
appended; its own query reads seven source positions. Its updated state,
not just its incoming message, produces the next vocabulary distribution.
The next pass looks up the new prefix's embeddings; it does not overwrite
lookup rows with the previous pass's contextual outputs.
The observed training target remains `.` for the original six-token prefix.

One all-parameter SGD step (learning rate 0.1) changes that target's probability
`.410236→.657754` and loss `.891023→.418925`. A separate all-position run
restarts the initial parameters and a trainable position table initialized to
zero: mean loss `2.339914→2.199261`. The last row's loss rises in this run
even though the mean decreases.

### Why values need their own role

The controlled value-only intervention keeps `X,W_Q,W_K` fixed and changes
`W_V` from `diag(1,2,1)` to `diag(0,2,1)`. Scores and attention weights
stay unchanged. The message becomes `[0,.571364,.285682]`, while the retained
input still gives updated state `[1,1.571364,.285682]`.
This is not a word replacement or an SGD step: changing a word could change
its key too. A second check sets the whole message to zero and verifies that
the residual output equals the original input.

Run in a Python environment with CPU PyTorch:

```bash
python lecture11/mega/bank_training.py --check
python lecture11/mega/bank_training.py --check --json lecture11/mega/bank-training-numbers.json
```

Or use the existing `uv` runtime without changing project dependencies:

```bash
uv run --with torch python lecture11/mega/bank_training.py --check
```

Checks cover manual/autograd agreement through the values, attention weights,
residual bypass and output map; finite differences across all seven parameter
groups; the value-only intervention; no embedding-table mutation in a forward
pass; fixed-parameter generation; causal equivalence; learned-position
gradients; and both SGD experiments.

The companion uses uppercase names for sequence matrices (`X,Q,K,V,A,H,Z,P`)
and `e_new` for one contextual embedding. Its JSON keeps legacy evidence keys
for compatibility: `x_new` means `e'`, `delta` means `Δe`, and `X_new`/`Delta`
mean the row-stacked `X'`/`ΔX`. These are data-export names, not additional
classroom notation. Saved numerical evidence is unchanged by the naming pass.

The contextual-update framing is informed by
[3Blue1Brown's attention explanation](https://www.3blue1brown.com/lessons/attention/).
We retain row-vector notation and explicitly separate the raw head message
`h`, mapped update `Δe=hW_O`, and contextual embedding `e'`.
The matrix walkthrough borrows the visual teaching approach of Visual AI's
linked video; see `video-vkhPtpUiLd8-notes.md`.
The earlier video-search analogy has been removed from L2. The worked bank
numbers and all technical diagrams are original, native Typst.

## Lecture-plan PDF

Pandoc converts the canonical Markdown outline to Typst; Typst renders the PDF.
The table filter reserves most of each row for the explanation.

```bash
pandoc lecture11/L11M-from-characters-to-transformers-outline.md \
  --from=markdown+tex_math_single_backslash+tex_math_dollars \
  --pdf-engine=typst \
  --metadata-file=lecture11/mega/plan-layout.yaml \
  --lua-filter=lecture11/mega/plan-tables.lua \
  -o output/pdf/from-characters-to-transformers-lecture-plan.pdf
```

## Parts

- `lecture1.typ`: NLP tasks → `aabid` → tokenization → natural-language LM
- `lecture2.typ`: fixed-window limits → attention → bank prediction/generation → computational graph/autograd → parameter update → causal training
- `lecture3.typ`: MHA → Transformer block → training → generation → synthesis
- `helpers.typ`: stable token, card, matrix, flow, and semantic-color helpers
