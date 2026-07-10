// Transformers II: BERT, GPT, Pretraining, Finetuning and Generation
// Lecture 16 · ES 667 Deep Learning
// Compile from the repo root:
//   typst compile --root . lecture16/L16-transformers-ii.typ /tmp/L16.pdf
//   typst compile --root . --input handout=true lecture16/L16-transformers-ii.typ /tmp/L16h.pdf
// Theme, palette, helpers and diagram builders live in common/metropolis.typ.

#import "../common/metropolis.typ": *
#import "../common/mldiag.typ": *
#show: metropolis-deck.with(
  title: [Transformers II: BERT, GPT, Pretraining, Finetuning and Generation],
  subtitle: [Same block, different mask + objective + head ⇒ different model family],
)

#let IA = "https://nipunbatra.github.io/interactive-articles/"
#let cream = rgb("#EFEEEB")

// ═══════════════════════════ fletcher diagrams (centerpieces) ═══════════════════════════

// ── (A) one picture, three families: encoder-only / decoder-only / encoder–decoder ──
#let threefam = align(center, diagram(spacing: (12mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rectn(pos, lbl, c, fill) = node(pos, align(center, text(size: 10pt, lbl)), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 6.5pt, stroke: 0.9pt + c, fill: fill)
  // Row 1 (top) — encoder-only (BERT)
  node((-1.35, -2), text(size: 11pt, fill: TEAL, weight: 600)[encoder-only], stroke: none, fill: none)
  rectn((0.4, -2), [tokens \ $x_1 dots x_T$], INK, cream)
  rectn((1.9, -2), [Encoder \ (bidirectional)], TEAL, TEAL.lighten(82%))
  rectn((3.5, -2), [contextual \ $h_1 dots h_T$], ACC, ACC.lighten(80%))
  edge((0.4, -2), (1.9, -2), "-|>", stroke: 0.8pt + MUTED); edge((1.9, -2), (3.5, -2), "-|>", stroke: 0.8pt + MUTED)
  // Row 2 (middle) — decoder-only (GPT)
  node((-1.35, 0), text(size: 11pt, fill: RED, weight: 600)[decoder-only], stroke: none, fill: none)
  rectn((0.4, 0), [prefix \ $x_1 dots x_t$], INK, cream)
  rectn((1.9, 0), [Decoder \ (causal)], RED, RED.lighten(85%))
  rectn((3.5, 0), [predict next \ $p(x_(t+1))$], ACC, ACC.lighten(80%))
  edge((0.4, 0), (1.9, 0), "-|>", stroke: 0.8pt + MUTED); edge((1.9, 0), (3.5, 0), "-|>", stroke: 0.8pt + MUTED)
  // Row 3 (bottom) — encoder–decoder (T5 / BART)
  node((-1.35, 2), text(size: 11pt, fill: BLUE, weight: 600)[enc–decoder], stroke: none, fill: none)
  rectn((0.4, 2), [source \ $x$], INK, cream)
  rectn((1.75, 2), [Encoder], TEAL, TEAL.lighten(82%))
  rectn((2.95, 2), [Decoder], RED, RED.lighten(85%))
  rectn((4.2, 2), [target \ $y$], ACC, ACC.lighten(80%))
  edge((0.4, 2), (1.75, 2), "-|>", stroke: 0.8pt + MUTED); edge((1.75, 2), (2.95, 2), "-|>", stroke: 0.8pt + MUTED); edge((2.95, 2), (4.2, 2), "-|>", stroke: 0.8pt + MUTED)
}))

// ── (B) cross-attention: Q from decoder, K,V from encoder ──
#let crossattn = align(center, diagram(spacing: (16mm, 11mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rectn(pos, lbl, c, fill) = node(pos, align(center, text(size: 10.5pt, lbl)), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: fill)
  rectn((0, 1.2), [encoder states \ $H$ #text(size: 9pt, fill: MUTED)[$(T times d)$]], TEAL, TEAL.lighten(82%))
  rectn((0, -1.2), [decoder state \ $S$ #text(size: 9pt, fill: MUTED)[$(U times d)$]], RED, RED.lighten(85%))
  rectn((2.5, 0), [$"softmax"((Q K^top) / sqrt(d)) thin V$], ACC, ACC.lighten(80%))
  rectn((4.9, 0), [context \ $c_u$], INK, cream)
  edge((0, 1.2), (2.5, 0), "-|>", stroke: 0.9pt + TEAL, label: text(size: 11pt, fill: TEAL)[$K, V$], label-side: left)
  edge((0, -1.2), (2.5, 0), "-|>", stroke: 0.9pt + RED, label: text(size: 11pt, fill: RED)[$Q$], label-side: left)
  edge((2.5, 0), (4.9, 0), "-|>", stroke: 0.9pt + ACC)
}))

// ── (C) pretrain → finetune flow ──
#let pretrainft = align(center, diagram(spacing: (14mm, 10mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rectn(pos, lbl, c, fill) = node(pos, align(center, text(size: 10pt, lbl)), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: fill)
  rectn((0, 0), [large corpus \ #text(size: 9pt, fill: MUTED)[unlabeled]], INK, cream)
  rectn((1.4, 0), [pretrain \ $-> theta_0$], TEAL, TEAL.lighten(82%))
  rectn((2.8, 0), [add task \ head], BLUE, BLUE.lighten(82%))
  rectn((4.2, 0), [finetune \ task data], ACC, ACC.lighten(80%))
  rectn((5.6, 0), [$theta^*$ \ task model], GREEN, GREEN.lighten(82%))
  for (a, b) in ((0, 1.4), (1.4, 2.8), (2.8, 4.2), (4.2, 5.6)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
}))

// ── (D) autoregressive generation loop ──
#let genloop = align(center, diagram(spacing: (14mm, 9mm), node-stroke: 0.9pt + INK, node-fill: white, {
  let rn(pos, lbl, c, fill) = node(pos, align(center, text(size: 10pt, lbl)), shape: fletcher.shapes.rect, corner-radius: 4pt, inset: 7pt, stroke: 0.9pt + c, fill: fill)
  rn((0, 0), [tokenize \ context], INK, cream)
  rn((1.2, 0), [forward \ pass], BLUE, BLUE.lighten(82%))
  rn((2.4, 0), [logits $z$], ACC, ACC.lighten(80%))
  rn((3.6, 0), [decode \ rule], TEAL, TEAL.lighten(82%))
  rn((4.8, 0), [append \ token], GREEN, GREEN.lighten(82%))
  for (a, b) in ((0, 1.2), (1.2, 2.4), (2.4, 3.6), (3.6, 4.8)) { edge((a, 0), (b, 0), "-|>", stroke: 0.8pt + MUTED) }
  edge((4.8, 0), (0, 0), "-|>", bend: -34deg, stroke: 0.9pt + GREEN)
  node((2.4, 1.05), text(size: 10pt, fill: GREEN)[repeat until `<EOS>` or max length], stroke: none, fill: white)
}))

#title-slide()

// ═══════════════════════════ PART I — Three families ═══════════════════════════
= One block, three families

== Where we are — recall Transformers I

Last lecture built the *Transformer block*: self-attention over all tokens, no recurrence.
#pause
$ Q = X W_Q, quad K = X W_K, quad V = X W_V $
#pause
$ "Attention"(Q, K, V) = "softmax"((Q K^top) / sqrt(d) + "mask") thin V $
#pause
#notebox[One block $=$ multi-head self-attention $-> $ residual $+$ norm $-> $ FFN $-> $ residual $+$ norm. This lecture: the *same block* becomes three different model *families*.]

== The whole lecture in one line

We keep the Transformer block *fixed* and change three knobs:
#pause
#align(center, text(size: 21pt, fill: INK)[
  *mask* #h(0.3em) $+$ #h(0.3em) *objective* #h(0.3em) $+$ #h(0.3em) *head* #h(0.5em) $=>$ #h(0.5em) *model family*
])
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 7pt), align: (left, left, left),
  table.header([*Knob*], [*Controls*], [*Choices*]),
  [attention mask], [what a token may *see*], [bidirectional / causal / cross],
  [pretraining objective], [what the model *learns to do*], [masked / next-token / denoise],
  [output head], [how we *read it out*], [`[CLS]` / LM head / decoder],
))
#pause
#result[architecture controls *information flow*; objective controls *what it learns*]

== The three families #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 7pt), align: (left, center, center, center),
  table.header([], [*Encoder-only*], [*Decoder-only*], [*Encoder–decoder*]),
  [mask], [bidirectional], [causal], [enc: bidir · dec: causal],
  [objective], [masked-token], [next-token], [seq2seq / denoise],
  [reads], [both sides], [left context], [source $->$ target],
  [best at], [understanding], [generation], [transduction],
  [example], [*BERT*], [*GPT*], [*T5 · BART*],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[three columns, one underlying block — the differences are the *mask*, the *objective* and the *head*])

== One picture, three families #V

#threefam
#pause
#align(center, text(size: 14pt, fill: MUTED)[encoder-only reads a whole sequence into *contextual reps*; decoder-only *predicts the next token*; encoder–decoder maps a *source* to a *target*])

== The masks that define them #V

The mask is the single biggest lever — it decides who may attend to whom:
#pause
#grid(columns: (1fr, 1fr, 1fr), gutter: 8pt,
  align(center + top)[
    #text(size: 12pt, weight: 600, fill: TEAL)[bidirectional (encoder)] \
    #text(size: 11pt, fill: MUTED)[BERT — no mask]
    #v(2.5mm)
    #attn-matrix(("1", "2", "3", "4", "5"), cell: 7mm,
      q-label: [query $i$], k-label: [key $j$])
  ],
  align(center + top)[
    #text(size: 12pt, weight: 600, fill: RED)[causal (decoder)] \
    #text(size: 11pt, fill: MUTED)[GPT — mask future]
    #v(2.5mm)
    #attn-matrix(("1", "2", "3", "4", "5"), mask: "causal", cell: 7mm,
      q-label: [query $i$], k-label: [key $j$])
  ],
  align(center + top)[
    #text(size: 12pt, weight: 600, fill: BLUE)[cross-attention] \
    #text(size: 11pt, fill: MUTED)[decoder $->$ encoder]
    #v(2.5mm)
    #attn-matrix((q: ("1", "2", "3", "4"), k: ("1", "2", "3", "4", "5")), cell: 7mm,
      q-label: [target $u$], k-label: [source $j$])
  ],
)
#pause
#align(center, text(size: 14pt, fill: MUTED)[bidirectional (BERT): all-to-all · causal (GPT): only the past · cross-attention: decoder reads encoder])

== Interactive: model-family masks #I

#interbox(link-to: IA + "attention")[
  Toggle between *bidirectional*, *causal* and *cross-attention* masks on a short sequence and watch which entries of the $T times T$ attention matrix light up — the same operation, three information-flow patterns.
]

// ═══════════════════════════ PART II — Encoder-only (BERT) ═══════════════════════════
= Encoder-only: BERT

== The encoder-only Transformer #D

Feed all tokens in at once; every token attends *every* non-padding token — *no causal mask*:
#pause
$ x_1, dots, x_T quad -->^"Encoder" quad h_1, dots, h_T $
#pause
- each $h_t$ is a *contextual representation* of token $t$ — informed by the whole sequence;
- no left-to-right restriction: position $t$ may look *left and right*.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the output is not a prediction — it is a *representation per token* to feed a task head])

== What BERT is good for #V

Encoder-only models shine at *understanding* — mapping text to a fixed target:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Task*], [*Shape*]),
  [sentence classification / sentiment], [$x_(1:T) -> y$],
  [natural language inference (NLI)], [pair $-> $ {entail, contradict, neutral}],
  [extractive QA], [passage $-> $ answer span],
  [token classification / NER], [$x_(1:T) -> y_(1:T)$],
  [retrieval / embeddings], [text $-> $ vector],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[whenever the output is a *label* or a *span*, both-side context helps])

== Why not next-token LM? #D

A causal LM only ever sees the *left*:
#pause
$ p(x_t | x_(<t)) quad quad #text(size: 15pt, fill: MUTED)[(left-to-right only)] $
#pause
But *understanding* a word often needs *both sides*:
#align(center, text(size: 17pt, fill: INK)[
  "The *bank* approved the loan" #h(0.6em) vs #h(0.6em) "The *bank* was flooded"
])
#pause
#alertbox[The word after `bank` disambiguates it. A left-only model has not seen it yet when it encodes `bank`. Bidirectional context lets the representation use *the whole sentence*.]

== Masked language modeling (MLM) #D

BERT's trick: *corrupt* some tokens, then predict the originals from *both sides*:
#pause
#codebox[`the cat sat on the [MASK]` #h(1em) $-> $ #h(1em) predict `mat`]
#pause
$ p_theta (x_m | x_(\\m)) quad quad #text(size: 15pt, fill: MUTED)[(recover masked token $m$ from all the others)] $
#pause
Loss is applied *only on the masked positions* $M$:
$ cal(L)_"MLM" = - sum_(m in M) log p_theta (x_m | tilde(x)) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[$tilde(x)$ is the corrupted sequence · unmasked tokens get *no* loss, they are only context])

== MLM vs next-token, side by side #V

#fig("/lecture16/figures/mlm_vs_clm.svg", w: 84%)
#pause
#align(center, text(size: 15pt, fill: MUTED)[MLM predicts a hidden token from *both* neighbours; a causal LM predicts the *next* token from the *left* only])

== MLM worked example #D

Sequence `deep learning is useful`; mask position 2:
#pause
#codebox[`deep [MASK] is useful` #h(1em) target: `learning`]
#pause
The encoder sees `deep`, `is`, `useful` (both sides) and scores the vocabulary at position 2:
$ cal(L) = - log p_theta ("learning" | tilde(x)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a single $V$-way classification at the masked slot — using *right* context (`is useful`) too])

== BERT input format #D

Two sentences packed into one sequence with special tokens:
#pause
#codebox[`[CLS]` sentence A `[SEP]` sentence B `[SEP]`]
#pause
Each token's input embedding is a *sum of three* embeddings:
$ r_t = e_t + p_t + s_t $
#pause
#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, center, left),
  table.header([*Component*], [*Symbol*], [*Encodes*]),
  [token], [$e_t$], [which word / subword],
  [position], [$p_t$], [where it sits],
  [segment], [$s_t$], [sentence A vs B],
))

== The `[CLS]` representation #D

A special `[CLS]` token sits at the front; its final state summarizes the *whole* sequence:
#pause
$ h_[#text[CLS]] quad -->^"classifier" quad p(y | x) = "softmax"(W thin h_[#text[CLS]] + b) $
#pause
#notebox[Attention lets `[CLS]` gather from every token, so its representation is a natural *pooled* summary. A small head on top of $h_[#text[CLS]]$ gives sentence-level predictions.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[classification, NLI, sentiment — all read out from this one vector])

== Token classification #D

Some tasks want a label at *every* position — put the head on *each* $h_t$:
#pause
$ p(y_t | x) = "softmax"(W thin h_t + b) quad quad #text(size: 15pt, fill: MUTED)[(one prediction per token)] $
#pause
- named-entity recognition, part-of-speech tagging;
- a *padding mask* keeps padded positions out of the loss.
#pause
#align(center, text(size: 16pt, fill: MUTED)[bidirectional context helps: tagging a word can use the words *after* it])

== Extractive QA #D

Answer $=$ a *span* in the passage. Learn a *start* and an *end* pointer over tokens:
#pause
$ p_"start"(t) = "softmax"(w_s^top h_t), quad quad p_"end"(t) = "softmax"(w_e^top h_t) $
#pause
Train to point at the true start $s^*$ and end $e^*$:
$ cal(L) = - log p_"start"(s^*) - log p_"end"(e^*) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[two little heads over the same contextual states — no text is generated, just *located*])

== The BERT finetuning pattern #V

Every downstream task is the *same* recipe:
#pause
#align(center, text(size: 18pt)[
  pretrained encoder #h(0.4em) $-> $ #h(0.4em) task head #h(0.4em) $-> $ #h(0.4em) supervised loss
])
#pause
#notebox[Start from the MLM-pretrained weights; bolt on a small task head ($h_[#text[CLS]]$ or per-token); train on labelled data. Typically *all* parameters update.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[one pretrained backbone, many cheap heads — the pretrain-then-finetune paradigm])

== RoBERTa: the recipe matters #D

Same architecture and MLM objective as BERT — but a *better training recipe*:
#pause
#two(
  notebox[*Changes* — more data, longer training, bigger batches, dynamic masking; dropped the next-sentence task.],
  notebox[*Result* — a large jump on the same benchmarks, with *no* architectural change.],
)
#pause
#result[the objective matters, but so does the *recipe*: data, scale, and training choices]

// ═══════════════════════════ PART III — Decoder-only (GPT) ═══════════════════════════
= Decoder-only: GPT

== The decoder-only Transformer #D

*Causal* self-attention: token $t$ may attend only to $x_(<= t)$, and predicts the next token:
#pause
$ h_t = f(x_(<= t)), quad quad p(x_(t+1) | x_(<= t)) = "softmax"(W_o thin h_t) $
#pause
#notebox[Drop the encoder and cross-attention from the Transformer: what remains is a stack of *causal self-attention $+$ FFN* blocks, capped by an LM head.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[this is the *same* next-token model of Lecture 11 — now with attention instead of a fixed window])

== The language-modeling objective #D

Factorize the sequence by the chain rule, then take negative log-likelihood:
#pause
$ p(x_(1:T)) = product_(t=1)^T p(x_t | x_(<t)) $
#pause
$ cal(L)_"LM" = - sum_(t=1)^T log p_theta (x_t | x_(<t)) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[token-level cross-entropy — *every* position is a training example, computed in parallel])

== The causal mask #D

Compute scores, then forbid attending to the future *before* the softmax:
#pause
$ A_(i j) = 0 quad #text(size: 15pt, fill: MUTED)[for] quad j > i $
#pause
#notebox[Set $S_(i j) = -oo$ when $j > i$; softmax turns those into $0$. Each row normalizes over the *past and present only* — no token can peek at its own target.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[this lower-triangular pattern is *the* difference between GPT and BERT attention])

== Shifted labels #D

Feed the sequence in; the target is the *same sequence shifted left by one*:
#pause
#align(center, table(
  columns: 5, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, center, center, center, center),
  table.header([position $t$], [1], [2], [3], [4]),
  [input $x_t$], [`<BOS>`], [`deep`], [`learning`], [`is`],
  [target $x_(t+1)$], [`deep`], [`learning`], [`is`], [`useful`],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[the causal mask makes this safe — position $t$ predicts $t+1$ without seeing it])

== GPT-style pretraining #D

Pretrain the decoder to predict the *next token* on a large corpus. Then adapt:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Adaptation*], [*What it means*]),
  [prompting], [condition on text; read the continuation],
  [finetuning], [continue training on task data],
  [instruction tuning], [finetune on (instruction $-> $ response) pairs],
  [RLHF #text(fill: MUTED)[(later)]], [align to human preferences],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[GPT-1 introduced *generative pretrain $+$ finetune*; the objective never changes — only the adaptation])

== GPT-2: LM as unsupervised multitask #D

Train a *large* decoder-only model on broad web text — then ask it to do tasks with *no* finetuning:
#pause
#notebox[A big enough next-token model performs many tasks *zero-shot*, purely from a prompt. Language modeling on diverse text is *implicitly* multitask learning (Radford et al.).]
#pause
#result[scale $+$ broad data $=>$ a single LM that generalizes to unseen tasks from prompts alone]

== Prompting as conditioning #D

Prompting is *not* an architecture change — it is just *conditioning the LM*:
#pause
$ p(#text[continuation] | #text[prompt]) $
#pause
#codebox[`Translate to French: cat =>` #h(0.6em) *(the model continues)* #h(0.4em) `chat`]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the prompt sets the left context; the *same* next-token machinery produces the answer])

== Few-shot prompting #D

Put a few *demonstrations* directly in the context, then the query:
#pause
#codebox[
`great movie => positive` \
`awful plot => negative` \
`loved every minute =>` #h(0.4em) *?*
]
#pause
#notebox[No weights change. The model *reads the pattern* from the examples in its context window and continues it — "in-context learning".]

== Interactive: in-context learning #I

#interbox(link-to: IA + "in-context-learning")[
  Add demonstrations to a prompt and watch a decoder-only model shift its next-token distribution toward the pattern you set — few-shot prompting with *no* gradient steps.
]

== Classification via prompting #D

Recast a label task as *next-token scoring*:
#pause
#codebox[`Review: a dull, lifeless film. Sentiment:` #h(0.4em) $-> $ #h(0.4em) `negative`]
#pause
Compare the LM's probability of each candidate label token:
$ hat(y) = arg max_(y in {"positive", "negative"}) thin p_theta (y | #text[prompt]) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[the classifier *is* the language model — pick the label it finds most probable])

== The generation loop #V

Decoder-only generation is one loop, run token by token:
#pause
#genloop
#pause
#align(center, text(size: 14pt, fill: MUTED)[tokenize $-> $ forward $-> $ logits $-> $ *decode rule* $-> $ append $-> $ repeat (the rule is Part VII)])

// ═══════════════════════════ PART IV — Encoder–decoder ═══════════════════════════
= Encoder–decoder

== The architecture #D

Two stacks: an encoder reads the *source*, a decoder generates the *target*:
#pause
$ H = "Encoder"(x), quad quad p(y_u | y_(<u), x) = "softmax"(W_o thin s_u) $
#pause
The decoder layer has *three* sublayers:
+ *causal* self-attention over the target prefix $y_(<u)$;
+ *cross-attention* into the encoder states $H$;
+ a position-wise FFN.
#pause
#align(center, text(size: 16pt, fill: MUTED)[self-attention reads what's generated so far; cross-attention reads the *source*])

== Cross-attention #D

The bridge between the stacks: *queries from the decoder*, *keys and values from the encoder*:
#pause
$ Q = S W_Q #h(0.4em) #text(size: 14pt, fill: MUTED)[(decoder)], quad K = H W_K, thin V = H W_V #h(0.4em) #text(size: 14pt, fill: MUTED)[(encoder)] $
#pause
$ "CrossAttn"(S, H) = "softmax"((Q K^top) / sqrt(d)) thin V $
#pause
#align(center, text(size: 16pt, fill: MUTED)[each target position *asks the source* which parts to attend to — this is how translation aligns])

== Cross-attention, drawn #V

#crossattn
#pause
#align(center, text(size: 14pt, fill: MUTED)[$Q$ comes from the decoder state $S$; $K$ and $V$ come from the encoder states $H$ — the decoder reads the source])

== Good for sequence-to-sequence #V

Encoder–decoder fits tasks where output length $U != $ input length $T$:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Task*], [*Source $-> $ target*]),
  [translation], [English sentence $-> $ German sentence],
  [summarization], [long document $-> $ short summary],
  [question answering], [question $+$ context $-> $ answer],
  [data-to-text], [table / triples $-> $ sentence],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[a full re-encoding of the source, then free-length generation of the target])

== T5: text-to-text #D

Cast *every* task as text $-> $ text, so one model and one loss handle all of them:
#pause
#codebox[
`translate English to German: That is good.` #h(0.4em) $-> $ #h(0.4em) `Das ist gut.` \
`sst2 sentence: a dull film` #h(0.4em) $-> $ #h(0.4em) `negative`
]
#pause
#notebox[Classification, translation, summarization — all become "read this text, write that text". A task prefix tells the model what to do (Raffel et al.).]

== BART: denoising seq2seq #D

*Corrupt* the input, then reconstruct the *original* — a bidirectional encoder feeds a causal decoder:
#pause
$ tilde(x) = C(x) quad -->^"Encoder–Decoder" quad x $
#pause
#two(
  notebox[*Encoder* — bidirectional, reads the corrupted text (like BERT).],
  notebox[*Decoder* — left-to-right, reconstructs the clean text (like GPT).],
)
#pause
#align(center, text(size: 15pt, fill: MUTED)[BART unifies BERT-style understanding and GPT-style generation in one seq2seq model (Lewis et al.)])

// ═══════════════════════════ PART V — Pretraining objectives compared ═══════════════════════════
= Pretraining objectives compared

== Three objectives #V

Each family is defined by *what it must predict*:
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 7pt), align: (left, left, left, center),
  table.header([*Objective*], [*Input $-> $ predict*], [*Architecture*], [*Family*]),
  [causal LM], [prefix $-> $ next token], [decoder-only], [GPT],
  [masked LM], [corrupted $-> $ masked tokens], [encoder-only], [BERT],
  [denoising seq2seq], [corrupted $-> $ original], [encoder–decoder], [T5 · BART],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[same block; the *prediction target* is what differs])

== The three losses #D

#pause
$ #text(fill: RED)[*causal LM*]: quad cal(L)_"CLM" = - sum_t log p(x_t | x_(<t)) $
#pause
$ #text(fill: TEAL)[*masked LM*]: quad cal(L)_"MLM" = - sum_(m in M) log p(x_m | tilde(x)) $
#pause
$ #text(fill: BLUE)[*denoising*]: quad p(x | tilde(x)) = product_t p(x_t | x_(<t), tilde(x)), quad tilde(x) = C(x) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[CLM $-> $ generation · MLM $-> $ understanding / bidirectional · denoising $-> $ reconstruct target from input])

== Span corruption (T5-style) #D

Instead of masking single tokens, mask *contiguous spans* and replace each with one sentinel:
#pause
#codebox[
input: #h(0.3em) `The quick brown fox jumps over the lazy dog` \
#v(2pt)
corrupt: #h(0.3em) `The quick <X> jumps over <Y> dog` \
#v(2pt)
target: #h(0.5em) `<X> brown fox <Y> the lazy`
]
#pause
#align(center, text(size: 15pt, fill: MUTED)[the decoder emits the *dropped spans*, tagged by sentinel — shorter targets, harder task])

== Which objective for which task #V

#align(center, table(
  columns: 3, stroke: 0.5pt + MUTED, inset: (x: 10pt, y: 7pt), align: (left, left, center),
  table.header([*You want*], [*Pick*], [*Why*]),
  [classify / tag / retrieve], [MLM (encoder)], [both-side context],
  [open-ended generation], [causal LM (decoder)], [native next-token],
  [translate / summarize], [seq2seq (enc–dec)], [source $-> $ target],
  [one model for everything], [text-to-text (T5)], [unified interface],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[match the *objective* to the *shape* of your task])

// ═══════════════════════════ PART VI — Finetuning & adaptation ═══════════════════════════
= Finetuning and adaptation

== Pretrain, then finetune #V

The dominant paradigm: pretrain once on a huge corpus, then specialize cheaply:
#pause
#pretrainft
#pause
$ theta_0 #h(0.3em) #text(size: 14pt, fill: MUTED)[(pretrained)] quad -> quad theta^* = arg min_theta thin cal(L)_"task"(theta) $
#pause
#align(center, text(size: 15pt, fill: MUTED)[the pretrained weights $theta_0$ are the *starting point*; finetuning nudges them to the task])

== Full finetuning #D

Update *all* parameters on the task data:
#pause
#two(
  notebox[*Pro* — best adaptation; the whole network can reshape to the task.],
  alertbox[*Con* — memory-heavy; can *overfit* small data; a *separate copy* of the model per task.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[the strong default when you have enough labelled data and compute])

== Feature extraction (frozen backbone) #D

*Freeze* the pretrained encoder; train *only* the task head:
#pause
$ underbrace(theta_"backbone", "frozen") quad + quad underbrace(theta_"head", "trained") $
#pause
- fast, cheap, tiny memory footprint;
- good for *small data* or a *quick baseline*;
- usually weaker than full finetuning.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the backbone is a fixed feature extractor; only a small head learns the task])

== PEFT preview #I

Between frozen and full: update a *tiny* set of extra parameters, keep the backbone fixed.
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Method*], [*Idea*]),
  [adapters], [small bottleneck modules inserted per layer],
  [prefix / prompt tuning], [learn virtual tokens prepended to the input],
  [LoRA], [learn low-rank updates $Delta W = B A$ to weights],
))
#pause
#interbox(link-to: IA + "lora-adapter")[
  Watch a low-rank update $Delta W = B A$ adapt a frozen weight matrix — a full lecture on *parameter-efficient* finetuning comes later.
]

== Classification-head example #D

Finetune BERT for sentiment — one small head over `[CLS]`:
#pause
#codebox[`[CLS] the movie was wonderful [SEP]`]
#pause
$ z = W thin h_[#text[CLS]] + b, quad quad cal(L) = - log "softmax"(z)_(y^*) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[pretrained encoder $+$ a linear head $+$ cross-entropy — the whole finetuning setup])

== Generation finetuning #D

For a generation task (e.g. summarization), finetune with the *seq2seq* loss:
#pause
$ cal(L) = - sum_(u=1)^U log p_theta (y_u | y_(<u), x) $
#pause
- $x$ is the document, $y$ the reference summary;
- teacher forcing: feed the true $y_(<u)$ during training.
#pause
#align(center, text(size: 16pt, fill: MUTED)[same next-token loss, now conditioned on the source $x$ — the model learns to *transduce*])

// ═══════════════════════════ PART VII — Decoding & generation ═══════════════════════════
= Decoding and generation

== The decoding problem

Given the next-token distribution $p_t$, *how do we choose* the next token?
#pause
#notebox[Training gives us $p_theta (x_(t+1) | x_(<= t))$. Decoding is a *separate* choice made at generation time — no weights change, but it hugely shapes the output.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[the same model can sound repetitive or creative depending only on the *decoding rule*])

== Greedy decoding #D

Always take the single most probable token:
#pause
$ hat(x)_(t+1) = arg max_v thin p_t (v) $
#pause
#two(
  notebox[*Pro* — simple, deterministic, fast; the model's local best guess.],
  alertbox[*Con* — repetitive and bland; locally-best $!=$ globally-best sequence.],
)

== Beam search #D

Keep the $B$ highest-scoring *partial sequences* at each step, by cumulative log-probability:
#pause
$ "score"(y_(1:u)) = sum_(k=1)^u log p(y_k | y_(<k)) $
#pause
- explores several hypotheses in parallel, not just one;
- *good* for translation / summarization (a single high-likelihood answer);
- *poor* for open-ended text — tends to be generic and repetitive.
#pause
#align(center, text(size: 16pt, fill: MUTED)[$B = 1$ is greedy; larger $B$ searches wider but costs more])

== Temperature #D

Rescale the logits by $tau$ *before* the softmax to control sharpness:
#pause
$ p_i (tau) = "softmax"(z \/ tau)_i $
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, left),
  table.header([$tau$], [*Effect*]),
  [$tau < 1$], [sharper — bolder, more deterministic],
  [$tau = 1$], [the model's own distribution],
  [$tau > 1$], [flatter — more random, more diverse],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[$tau -> 0$ recovers greedy; $tau -> oo$ approaches uniform])

== Top-k and nucleus (top-p) #D

Truncate the tail *before* sampling, so rare tokens can't derail generation:
#pause
#two(
  notebox[*Top-$k$* — keep the $k$ most probable tokens, renormalize, sample.],
  notebox[*Nucleus (top-$p$)* — keep the smallest set $S$ with $sum_(i in S) p_i >= p$, then sample.],
)
#pause
#align(center, text(size: 16pt, fill: MUTED)[top-$p$ *adapts* the candidate set to the model's confidence — narrow when sure, wide when unsure])

== Repetition penalties #D

Explicit fixes for the degenerate repetition that sampling and greedy can produce:
#pause
- *no-repeat n-gram* — forbid emitting an n-gram already seen;
- *repetition / frequency penalty* — down-weight tokens already generated;
- *presence penalty* — discourage reusing any token that has appeared.
#pause
#align(center, text(size: 16pt, fill: MUTED)[cheap, decoding-time nudges away from loops — no retraining])

== Interactive: the decoding lab #I

#interbox(link-to: IA + "softmax-temperature")[
  Sweep *temperature*, *top-$k$* and *top-$p$* on a fixed set of logits and watch the next-token bar chart sharpen or flatten — then sample repeatedly and see the diversity of outputs change.
]

// ═══════════════════════════ PART VIII — Worked mini examples ═══════════════════════════
= Worked examples

== BERT MLM: the loss #D

Sentence `I like [MASK]`; candidates and BERT's predicted probabilities:
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, center, center, center),
  table.header([token], [`cats`], [`blue`], [`running`]),
  [$p$], [*0.80*], [0.05], [0.15],
))
#pause
Target is `cats`; cross-entropy picks out its probability:
$ cal(L) = - log(0.80) approx 0.223 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[a confident, correct prediction — small loss; a perfect one would give $0$])

== GPT next-token: the loss #D

Prompt `I like`; the decoder's next-token distribution:
#pause
#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (center, center, center, center),
  table.header([token], [`cats`], [`dogs`], [`blue`]),
  [$p$], [*0.6*], [0.3], [0.1],
))
#pause
Target is `cats`:
$ cal(L) = - log(0.6) approx 0.511 $
#pause
#align(center, text(size: 16pt, fill: MUTED)[same cross-entropy loss as MLM — the difference is *causal* context, not left-and-right])

== T5 text-to-text: the loss #D

Prompt the encoder, generate the target with the decoder:
#pause
#codebox[`translate English to Hindi: I like cats` #h(0.4em) $-> $ #h(0.4em) target sequence]
#pause
Sum the per-token negative log-likelihood over the target, conditioned on the source:
$ cal(L) = - sum_(u=1)^U log p_theta (y_u | y_(<u), x) $
#pause
#align(center, text(size: 16pt, fill: MUTED)[exactly the seq2seq loss — the *source* $x$ conditions every target-token prediction])

// ═══════════════════════════ PART IX — Practical model choice ═══════════════════════════
= Choosing a model in practice

== Classification #D

Three valid routes — pick by data and infrastructure:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Approach*], [*How*]),
  [encoder-only], [`[CLS]` $-> $ classifier head],
  [decoder-only], [prompt $-> $ score label tokens],
  [encoder–decoder], [input $-> $ generate label text],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[encoder-only is the classic strong, cheap default for a fixed label set])

== Generation #D

#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Need*], [*Choice*]),
  [open-ended text], [decoder-only, sampling (top-$p$)],
  [input-conditioned output], [encoder–decoder],
  [translation], [beam search (high-likelihood answer)],
  [creative / diverse], [sampling with temperature],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[decode *deterministically* when there's one right answer; *sample* when diversity matters])

== Token tagging and small data #D

#two(
  notebox[*Token tagging* — encoder-only. Both-side context helps label each token: $h_t -> y_t$ (NER, POS).],
  notebox[*Small data* — start from a *pretrained* model; escalate frozen $-> $ PEFT $-> $ full only as needed.],
)
#pause
#result[always start pretrained; keep a *held-out validation set* and watch for overfitting]

// ═══════════════════════════ PART X — Limitations ═══════════════════════════
= Limitations

== Context length and cost #D

Self-attention is *quadratic* in sequence length, and context is *finite*:
#pause
$ O(T^2 d) #text(size: 15pt, fill: MUTED)[ compute], quad quad T <= T_"max" $
#pause
#alertbox[Every token attends every token, so cost grows with $T^2$. Models have a hard context limit $T_"max"$; long inputs must be truncated or chunked.]
#pause
#align(center, text(size: 16pt, fill: MUTED)[efficient / long-context attention is a whole topic for a later lecture])

== Tokenization issues #D

The tokenizer silently shapes everything downstream:
#pause
- text is split into *subwords* (BPE) — not words or characters;
- *script differences* change how many tokens a language needs;
- sequence length in tokens depends on the tokenizer, not just the text.
#pause
#alertbox[*Perplexity is not comparable across tokenizers* — a character model and a subword model predict different-sized units. Never compare PPL across vocabularies.]

== Worked BPE: learn one subword #D

Start from characters. In a toy corpus with #raw("low") three times and #raw("lower") once:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 12pt, y: 6pt), align: (left, left),
  table.header([*word*], [*initial symbols*]),
  [#raw("low") × 3], [#raw("l o w </w>")],
  [#raw("lower") × 1], [#raw("l o w e r </w>")],
))
#pause
The adjacent pairs #raw("(l, o)") and #raw("(o, w)") each occur four times. Choose one tie arbitrarily:
$l o -> "lo"$, then the next frequent pair can give $"lo" w -> "low"$.
#pause
#result[after those two merges, an unseen #raw("lowest") can begin as #raw("[low, e, s, t, </w>]") — frequent pieces are reused.]
#pause
#align(center, text(size: 15pt, fill: MUTED)[BPE discovers frequent character sequences, not guaranteed linguistic morphemes.])

== Interactive: tokenization #I

#interbox(link-to: IA + "bpe-merges")[
  Watch *byte-pair encoding* merge frequent character pairs into subword tokens — and see how the same sentence becomes a different number of tokens under different vocabularies.
]

== Pretraining data matters #D

A pretrained model inherits its *corpus*:
#pause
- *biases* and toxicity present in the data;
- *coverage* — domains, languages, styles it did or didn't see;
- *factual staleness* — knowledge frozen at the training cutoff.
#pause
#align(center, text(size: 16pt, fill: MUTED)[the model is a mirror of its data — know what it was trained on before you trust it])

== Evaluation depends on the task #V

No single number fits every model family:
#pause
#align(center, table(
  columns: 2, stroke: 0.5pt + MUTED, inset: (x: 11pt, y: 6pt), align: (left, left),
  table.header([*Task type*], [*Metrics*]),
  [classification], [accuracy · F1 · AUROC],
  [generation], [BLEU · ROUGE · BERTScore · human],
  [language modeling], [NLL · perplexity],
  [retrieval], [MRR · Recall\@k · NDCG],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[choose the metric that matches the *task shape*, not the architecture])

// ═══════════════════════════ PART XI — Summary ═══════════════════════════
= Summary

== The model families, one table #V

#align(center, table(
  columns: 4, stroke: 0.5pt + MUTED, inset: (x: 9pt, y: 7pt), align: (left, center, center, center),
  table.header([], [*Encoder-only*], [*Decoder-only*], [*Encoder–decoder*]),
  [mask], [bidirectional], [causal], [bidir $+$ causal],
  [pretraining], [masked LM], [next-token], [denoise / seq2seq],
  [best fit], [understanding], [generation], [transduction],
  [example], [BERT · RoBERTa], [GPT], [T5 · BART],
))
#pause
#align(center, text(size: 16pt, fill: MUTED)[mask $times$ pretraining objective $times$ head $=>$ the family and its sweet spot])

== The objectives, in one breath #D

#pause
- *BERT* — predict the *missing tokens* using *both sides* (understanding);
#pause
- *GPT* — predict the *next token* from the *previous* ones (generation);
#pause
- *T5 / BART* — *reconstruct or generate* the target from a corrupted / source input (transduction).
#pause
#result[one Transformer block; three ways to train it, three things it learns to do]

== Final mental model — three knobs

#align(center, text(size: 22pt)[
  *Architecture* controls *information flow* — who attends to whom.
  #v(6pt)
  *Objective* controls *what it learns to do* — predict, mask, or reconstruct.
  #v(6pt)
  *Finetuning / prompting* controls *task adaptation* — how we specialize it.
  #v(8pt)
  #text(size: 18pt, fill: MUTED)[mask $+$ objective $+$ head $=>$ model family]
])

#focus-slide[
  Same block — three families, by *mask*, *objective* and *head*.
  #v(12pt)
  #set text(size: 22pt)
  Next: *Vision Transformers & multimodal* — images as patches, ViT, and CLIP.
]
