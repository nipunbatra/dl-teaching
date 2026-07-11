---
title: "Lecture 17: Transformers II"
source: "https://chatgpt.com/share/6a50728a-9720-83ee-a33d-5f2e3ff744fe"
status: "Imported from the finalized shared-course discussion"
---

# Lecture 17: Transformers II  
## BERT, GPT, Pretraining, Finetuning and Generation

Core story:

\[
\boxed{
\text{Same Transformer block, different mask/objective} \Rightarrow \text{different model families.}
}
\]

BERT uses bidirectional encoder pretraining with masked language modelling; GPT-style models use decoder-only causal language modelling; T5 converts NLP tasks into a unified text-to-text format.

---

# Part I — Three Transformer families

### Slide 1 — Title

**Transformers II**

> BERT, GPT, pretraining, finetuning and generation

---

### Slide 2 — Recall from Transformers I

\[
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
\]

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt d}+\text{mask}
\right)V
\]

Today:

\[
\text{mask}+\text{objective}+\text{head}
\Rightarrow
\text{model family}
\]

---

### Slide 3 — Encoder-only, decoder-only, encoder–decoder

| Family | Attention | Objective | Example |
|---|---|---|---|
| Encoder-only | bidirectional | masked token prediction | BERT |
| Decoder-only | causal | next-token prediction | GPT |
| Encoder–decoder | encoder bidir + decoder causal | seq2seq | T5/BART |

---

### Slide 4 — One picture

```text
Encoder-only:      x1 x2 x3 x4  → contextual reps
Decoder-only:      x1 x2 x3 x4  → predict next tokens
Encoder-decoder:   source → encoder → decoder → target
```

---

# Part II — Encoder-only models: BERT

### Slide 5 — Encoder-only Transformer

Input tokens:

\[
x_1,\ldots,x_T
\]

Output contextual representations:

\[
h_1,\ldots,h_T.
\]

Each token can attend to every other non-padding token.

No causal mask.

---

### Slide 6 — What is BERT good for?

Tasks where full input is available:

- classification;
- sentiment;
- natural language inference;
- extractive QA;
- token classification;
- named entity recognition;
- retrieval/reranking.

BERT is designed to pretrain deep bidirectional representations by conditioning on both left and right context.

---

### Slide 7 — Why not standard next-token LM for BERT?

Next-token LM only conditions left-to-right:

\[
p(x_t\mid x_{<t}).
\]

For understanding tasks, we often want:

\[
h_t=f(x_1,\ldots,x_T)
\]

using both sides.

Example:

```text
The bank approved the loan.
The bank was flooded.
```

Right context matters.

---

### Slide 8 — Masked Language Modelling

Corrupt some tokens:

```text
the cat sat on the [MASK]
```

Predict original token:

\[
p_\theta(x_m\mid x_{\setminus m}).
\]

Loss only on masked positions:

\[
L_{\text{MLM}}
=
-\sum_{m\in M}
\log p_\theta(x_m\mid \tilde x).
\]

---

### Slide 9 — MLM worked example

Original:

```text
deep learning is useful
```

Corrupted:

```text
deep [MASK] is useful
```

Target at masked position:

```text
learning
```

Model output at mask position:

\[
p(\cdot\mid \texttt{deep [MASK] is useful})
\]

Loss:

\[
-\log p(\texttt{learning})
\]

---

### Slide 10 — BERT input format

\[
[\texttt{CLS}],\ \text{sentence A},\ [\texttt{SEP}],\ \text{sentence B},\ [\texttt{SEP}]
\]

Embedding:

\[
r_t
=
e_t+p_t+s_t
\]

where:

- \(e_t\): token embedding;
- \(p_t\): position embedding;
- \(s_t\): segment/token-type embedding.

---

### Slide 11 — `[CLS]` representation

For classification:

\[
h_{\texttt{CLS}}
\]

is fed to classifier:

\[
p(y\mid x)=\operatorname{softmax}(Wh_{\texttt{CLS}}+b).
\]

Loss:

\[
L=-\log p(y).
\]

---

### Slide 12 — Token classification

For NER/POS tagging:

\[
h_t\rightarrow p(y_t\mid x)
\]

\[
L=-\sum_t\log p(y_t).
\]

Use padding mask.

---

### Slide 13 — Extractive QA

Given context and question, predict answer span:

\[
p_{\text{start}}(t)=\operatorname{softmax}(w_s^\top h_t)
\]

\[
p_{\text{end}}(t)=\operatorname{softmax}(w_e^\top h_t)
\]

Loss:

\[
L=-\log p_{\text{start}}(s^\star)-\log p_{\text{end}}(e^\star).
\]

---

### Slide 14 — BERT finetuning pattern

```text
pretrained encoder
        ↓
task-specific head
        ↓
supervised loss
```

All parameters usually update:

\[
\theta_{\text{encoder}},\theta_{\text{head}}.
\]

---

### Slide 15 — RoBERTa lesson

RoBERTa showed that BERT-style pretraining is very sensitive to recipe choices: training longer, on more data, with larger batches and modified masking/NSP choices can substantially improve performance.

Teaching point:

\[
\boxed{\text{pretraining objective matters, but training recipe also matters.}}
\]

---

# Part III — Decoder-only models: GPT

### Slide 16 — Decoder-only Transformer

Input:

\[
x_1,\ldots,x_T
\]

Causal self-attention:

\[
h_t=f(x_{\le t})
\]

Predict:

\[
p(x_{t+1}\mid x_{\le t}).
\]

---

### Slide 17 — Language-model factorization

\[
p(x_{1:T})
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
\]

Training loss:

\[
L_{\text{LM}}
=
-\sum_t\log p_\theta(x_t\mid x_{<t}).
\]

Same as earlier next-token lecture, now with Transformer backbone.

---

### Slide 18 — Causal mask

\[
A_{ij}=0\quad\text{if }j>i.
\]

At position \(i\), token can attend to:

\[
1,\ldots,i
\]

not future positions.

---

### Slide 19 — Shifted labels

Input:

```text
<BOS> deep learning is
```

Target:

```text
deep learning is useful
```

Logits at position \(t\) predict token \(t+1\).

---

### Slide 20 — GPT-style pretraining

Train on large text corpus using next-token prediction.

\[
\min_\theta
-\sum_t\log p_\theta(x_t\mid x_{<t})
\]

Then adapt to tasks via:

- prompting;
- finetuning;
- instruction tuning;
- RLHF/direct preference optimization later.

GPT-1 explored generative pretraining followed by supervised finetuning for language understanding tasks.

---

### Slide 21 — GPT-2 lesson

GPT-2 demonstrated that a large decoder-only LM trained on broad web text could perform several tasks in a zero-shot manner from natural-language prompts, strengthening the idea of language modelling as unsupervised multitask learning.

---

### Slide 22 — Prompting as conditioning

Prompt:

```text
Translate English to Hindi:
English: I like cats.
Hindi:
```

Model estimates:

\[
p(\text{continuation}\mid \text{prompt})
\]

No architectural change.

---

### Slide 23 — Few-shot prompting

Prompt includes demonstrations:

```text
English: good morning
Hindi: सुप्रभात

English: thank you
Hindi: धन्यवाद

English: how are you
Hindi:
```

Model uses context examples.

---

### Slide 24 — Decoder-only classification via prompting

Sentiment prompt:

```text
Review: The movie was excellent.
Sentiment:
```

Possible next tokens:

\[
\texttt{positive},\quad \texttt{negative}.
\]

Classification can be recast as next-token scoring.

---

### Slide 25 — Decoder-only generation loop

1. Tokenize prompt.
2. Forward pass.
3. Get logits for next token.
4. Apply decoding rule.
5. Append sampled token.
6. Repeat.

\[
x_{t+1}\sim p_\theta(\cdot\mid x_{\le t})
\]

---

# Part IV — Encoder–decoder Transformers

### Slide 26 — Encoder–decoder architecture

Encoder:

\[
H=\operatorname{Encoder}(x_{1:T})
\]

Decoder:

\[
p(y_u\mid y_{<u},x)
=
\operatorname{softmax}(W_os_u)
\]

Decoder uses:

- causal self-attention over target prefix;
- cross-attention over encoder states.

---

### Slide 27 — Cross-attention

Queries from decoder:

\[
Q=S W_Q
\]

Keys and values from encoder:

\[
K=H W_K,\quad V=H W_V.
\]

\[
\operatorname{CrossAttn}(S,H)
=
\operatorname{softmax}\left(
\frac{QK^\top}{\sqrt d}
\right)V.
\]

---

### Slide 28 — Good for seq2seq tasks

- translation;
- summarization;
- question answering;
- data-to-text;
- speech/text transduction;
- text simplification.

Input and output lengths may differ:

\[
T\
eq U.
\]

---

### Slide 29 — T5 text-to-text framing

T5 converts every task into:

\[
\text{text input}\rightarrow\text{text output}.
\]

Examples:

```text
translate English to German: That is good.
→ Das ist gut.
```

```text
sst2 sentence: this movie is great
→ positive
```

T5 systematically studied transfer learning with a unified text-to-text Transformer framework.

---

### Slide 30 — BART denoising seq2seq

Corrupt text:

```text
The [MASK] sat on the mat.
```

Reconstruct original:

```text
The cat sat on the mat.
```

BART trains a seq2seq model to reconstruct original text from corrupted text, combining a bidirectional encoder with a left-to-right decoder.

---

# Part V — Pretraining objectives compared

### Slide 31 — Three objectives

| Objective | Input | Target | Model family |
|---|---|---|---|
| Causal LM | prefix | next token | decoder-only |
| MLM | corrupted text | masked tokens | encoder-only |
| Denoising seq2seq | corrupted text | original text | encoder–decoder |

---

### Slide 32 — Causal LM loss

\[
L_{\text{CLM}}
=
-\sum_t\log p_\theta(x_t\mid x_{<t}).
\]

Good for:

- generation;
- prompting;
- autoregressive modelling.

---

### Slide 33 — MLM loss

\[
L_{\text{MLM}}
=
-\sum_{m\in M}\log p_\theta(x_m\mid \tilde x).
\]

Good for:

- understanding;
- classification;
- token-level tasks;
- bidirectional representations.

---

### Slide 34 — Denoising seq2seq loss

Corrupt:

\[
\tilde x=C(x)
\]

Reconstruct:

\[
p_\theta(x\mid \tilde x)
=
\prod_t p_\theta(x_t\mid x_{<t},\tilde x).
\]

Loss:

\[
L=-\sum_t\log p_\theta(x_t\mid x_{<t},\tilde x).
\]

Good for generation conditioned on input.

---

### Slide 35 — Span corruption

Instead of masking individual tokens, mask spans:

```text
The quick brown fox jumps over the lazy dog.
```

becomes:

```text
The quick <X> jumps over <Y> dog.
```

Target:

```text
<X> brown fox <Y> the lazy
```

This is central to T5-style denoising.

---

### Slide 36 — Which objective for which task?

| Task | Better default |
|---|---|
| sentiment classification | encoder-only / decoder prompting |
| extractive QA | encoder-only |
| summarization | encoder–decoder or decoder-only |
| translation | encoder–decoder |
| free-form generation | decoder-only |
| NER/POS | encoder-only |
| chat | decoder-only or encoder–decoder |

---

# Part VI — Finetuning and adaptation

### Slide 37 — Pretrain then finetune

\[
\theta_0
=
\text{pretrained weights}
\]

\[
\theta^\star
=
\arg\min_\theta
L_{\text{task}}(\theta)
\]

Pretraining learns general representations. Finetuning adapts them.

---

### Slide 38 — Full finetuning

Update all parameters:

\[
\theta_{\text{all}}\leftarrow
\theta_{\text{all}}-\eta\
abla_\theta L.
\]

Pros:

- best task adaptation.

Cons:

- memory expensive;
- can overfit small data;
- separate copy per task.

---

### Slide 39 — Feature extraction / frozen encoder

Freeze backbone:

\[
\theta_{\text{base}}\ \text{fixed}
\]

Train only head:

\[
\theta_{\text{head}}.
\]

Useful for small data, quick baselines.

---

### Slide 40 — Parameter-efficient finetuning preview

Instead of updating all weights, train small additional modules or low-rank updates.

Examples:

- adapters;
- prefix/prompt tuning;
- LoRA.

Do only as preview; detailed modern adaptation can be a later lecture.

---

### Slide 41 — Classification head example

Input:

```text
[CLS] movie was wonderful [SEP]
```

BERT output:

\[
h_{\texttt{CLS}}\in\mathbb R^d.
\]

Head:

\[
z=Wh_{\texttt{CLS}}+b.
\]

Loss:

\[
L=-\log p(y).
\]

---

### Slide 42 — Sequence generation finetuning

For summarization:

Input:

\[
x=\text{document}
\]

Target:

\[
y=\text{summary}.
\]

Objective:

\[
L=-\sum_u\log p_\theta(y_u\mid y_{<u},x).
\]

---

# Part VII — Decoding and generation

### Slide 43 — Greedy decoding

\[
x_{t+1}=\arg\max_v p(v\mid x_{\le t}).
\]

Pros:

- simple;
- deterministic.

Cons:

- repetitive;
- low diversity;
- locally greedy.

---

### Slide 44 — Beam search

Maintain top \(B\) hypotheses by cumulative score:

\[
S(y_{1:t})=\sum_{j=1}^t\log p(y_j\mid y_{<j},x).
\]

Good for tasks like translation, but less ideal for open-ended creative generation.

---

### Slide 45 — Temperature sampling

\[
p_i(\tau)
=
\frac{\exp(z_i/\tau)}
{\sum_j\exp(z_j/\tau)}.
\]

\[
\tau<1:\text{ sharper}
\]

\[
\tau>1:\text{ flatter}
\]

---

### Slide 46 — Top-\(k\) sampling

Keep only top \(k\) tokens:

\[
\tilde p_i
=
\frac{p_i\mathbf 1[i\in\text{top-}k]}
{\sum_{j\in\text{top-}k}p_j}.
\]

---

### Slide 47 — Nucleus / top-\(p\) sampling

Choose smallest set \(S\) such that:

\[
\sum_{i\in S}p_i\ge p.
\]

Then sample from \(S\).

This adapts the candidate set size to confidence.

---

### Slide 48 — Repetition penalties

Problems:

```text
the the the the
```

Mitigations:

- no-repeat n-gram;
- repetition penalty;
- frequency penalty;
- presence penalty;
- better training/instruction data.

---

### Slide 49 — Interactive 1: decoding lab

`decoding-lab.html`

Controls:

- logits;
- temperature;
- top-\(k\);
- top-\(p\);
- repetition penalty;
- greedy/beam/sample.

Show:

- final distribution;
- sampled token;
- generated sequence.

---

# Part VIII — Worked mini examples

### Slide 50 — BERT MLM example

Input:

```text
I like [MASK]
```

Vocabulary:

\[
[\texttt{cats},\texttt{blue},\texttt{running}]
\]

Model probabilities:

\[
[0.80,0.05,0.15].
\]

Target:

\[
\texttt{cats}.
\]

Loss:

\[
-\log(0.80)=0.223.
\]

---

### Slide 51 — GPT next-token example

Prompt:

```text
I like
```

Probabilities:

\[
p(\texttt{cats})=0.6,
\quad
p(\texttt{dogs})=0.3,
\quad
p(\texttt{blue})=0.1.
\]

If target is `cats`:

\[
L=-\log(0.6)=0.511.
\]

---

### Slide 52 — T5 text-to-text example

Input:

```text
translate English to Hindi: I like cats
```

Target:

```text
मुझे बिल्लियाँ पसंद हैं
```

Loss:

\[
-\sum_u
\log p(y_u\mid y_{<u},x).
\]

---

# Part IX — Practical model choice

### Slide 53 — If your task is classification

Options:

1. Encoder-only:
   \[
   [\texttt{CLS}]\rightarrow\text{classifier}
   \]

2. Decoder-only:
   \[
   \text{prompt}\rightarrow\text{label token}
   \]

3. Encoder–decoder:
   \[
   \text{input}\rightarrow\text{label text}
   \]

---

### Slide 54 — If your task is generation

Use:

- decoder-only for open-ended continuation/chat;
- encoder–decoder for input-conditioned generation like translation/summarization;
- beam search for constrained translation-style tasks;
- sampling for open-ended generation.

---

### Slide 55 — If your task is token tagging

Usually:

\[
\text{encoder-only}
\]

because each token benefits from both left and right context.

Example:

\[
h_t\rightarrow y_t.
\]

---

### Slide 56 — If your data is small

Start with pretrained model.

Try:

1. frozen encoder/head;
2. full finetuning;
3. parameter-efficient tuning;
4. prompting if using decoder-only LMs.

Validation discipline matters.

---

# Part X — Limitations and caveats

### Slide 57 — Context length

Transformers have finite context windows:

\[
T_{\max}.
\]

Self-attention cost:

\[
O(T^2d).
\]

Long-context methods modify attention or memory, but this is a later topic.

---

### Slide 58 — Tokenization issues

Model works over tokens, not words.

Consequences:

- rare words split into subwords;
- languages/scripts behave differently;
- sequence length depends on tokenizer;
- perplexity not directly comparable across tokenizers.

---

### Slide 59 — Pretraining data matters

Pretrained models inherit:

- corpus biases;
- domain coverage;
- style;
- factual staleness;
- noise.

Model behaviour is not only architecture/objective.

---

### Slide 60 — Evaluation depends on task

Classification:

\[
\text{accuracy/F1/AUROC}
\]

Generation:

\[
\text{BLEU/ROUGE/BERTScore/human eval}
\]

Language modelling:

\[
\text{NLL/perplexity}
\]

Retrieval/ranking:

\[
\text{MRR/Recall@k/NDCG}
\]

---

# Part XI — Summary

### Slide 61 — Model family summary

| Family | Attention mask | Pretraining | Best fit |
|---|---|---|---|
| Encoder-only | bidirectional | MLM | understanding |
| Decoder-only | causal | next-token LM | generation/prompting |
| Encoder–decoder | encoder bidir, decoder causal | denoising/text-to-text | conditional generation |

---

### Slide 62 — Objective summary

\[
\boxed{
\text{BERT: predict missing tokens from both sides}
}
\]

\[
\boxed{
\text{GPT: predict next token from previous tokens}
}
\]

\[
\boxed{
\text{T5/BART: reconstruct/generate target text from input text}
}
\]

---

### Slide 63 — Final mental model

\[
\boxed{
\text{Architecture controls information flow.}
}
\]

\[
\boxed{
\text{Objective controls what the model learns to do.}
}
\]

\[
\boxed{
\text{Finetuning/prompting controls task adaptation.}
}
\]

---

### Slide 64 — Next lecture

**Vision Transformers and Multimodal Models**

Topics:

- images as patches;
- ViT;
- patch embeddings;
- CLS token for vision;
- multimodal contrastive learning;
- CLIP-style models.

---

# Timing

| Section | Slides | Time |
|---|---:|---:|
| Families overview | 1–4 | 5 min |
| BERT / encoder-only | 5–15 | 14 min |
| GPT / decoder-only | 16–25 | 13 min |
| Encoder–decoder | 26–30 | 8 min |
| Pretraining objectives | 31–36 | 10 min |
| Finetuning/adaptation | 37–42 | 8 min |
| Decoding | 43–49 | 10 min |
| Worked examples | 50–52 | 5 min |
| Practical choice/caveats | 53–60 | 9 min |
| Summary | 61–64 | 3 min |

---

# Interactives

1. `model-family-masks.html`  
   Encoder bidirectional mask, decoder causal mask, encoder–decoder cross-attention.

2. `mlm-bert-demo.html`  
   Mask positions, predict masked tokens, compute MLM loss.

3. `gpt-next-token-demo.html`  
   Causal LM shifted labels and token-by-token loss.

4. `text-to-text-demo.html`  
   T5/BART style input-target construction.

5. `decoding-lab.html`  
   Greedy, beam, temperature, top-\(k\), top-\(p\).

6. `finetuning-heads.html`  
   Classification head, token head, generation head.

---

# Notebooks

1. `01_transformer_masks_families.ipynb`
2. `02_mlm_objective_from_scratch.ipynb`
3. `03_causal_lm_loss_shifted_labels.ipynb`
4. `04_text_to_text_data_format.ipynb`
5. `05_decoding_strategies.ipynb`
6. `06_finetune_encoder_classifier.ipynb`
7. `07_finetune_decoder_prompting.ipynb`
