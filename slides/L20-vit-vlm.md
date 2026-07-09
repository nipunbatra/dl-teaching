---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Vision Transformers & Vision-Language Models

## Lecture 20 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar*

---

# The whole lecture, in one idea

A network doesn't care whether a token came from a sentence or a photo. Cut an image into **patches**, embed each as a **token**, and the *same* Transformer — the same attention from L14 — now **sees**.

<div class="keypoint">

**Vision is just more tokens.**
Three moves, each a tiny modification of machinery you already own:
**ViT** turns patches into tokens · **CLIP** puts image and text tokens in **one shared space** (L19's contrastive loss, across modalities) · **LLaVA** lets an LLM read image tokens like words.

</div>

You built attention in L14, Transformers in L15, LLMs in L16–L17, and contrastive learning in L19. Today those pieces **compose** into a model that sees and talks.

$$\underbrace{\text{patches}\to\text{tokens}}_{\text{ViT}} \;\rightarrow\; \underbrace{\text{one shared space}}_{\text{CLIP}} \;\rightarrow\; \underbrace{\text{LLM reads image tokens}}_{\text{LLaVA}}$$

---

# How today connects to the rest of the course

<div class="columns">
<div>

**Multimodal AI is three prior lectures, composed:**
- **L14–L15** — attention & the Transformer: *the* engine, unchanged, the whole way through
- **L19** — contrastive learning: CLIP is L19's pull-together / push-apart loss, run across **image + text** instead of two crops
- **L15–L17** — the LLM: the "brain" LLaVA gives eyes to

</div>
<div>

**The payoff of each part, in one line:**
- **ViT** — an image *is* a sequence of tokens
- **CLIP** — a **caption is a classifier**
- **LLaVA** — **vision is just more tokens** for the LLM

</div>
</div>

<div class="insight">

In Ng's spirit, we don't memorize four new architectures — we build one idea end-to-end from parts you already have, and watch it generalize.

</div>

---

<!-- _class: section-divider -->

## Part 1 · ViT — patches are tokens

---

# ViT in one sentence

<div class="keypoint">

A **Vision Transformer** cuts the image into 16×16 patches, linearly embeds each patch into one token, and feeds the sequence to a **plain Transformer** — the exact attention stack from L14–L15, with **no convolutions at all**.

</div>

The whole trick is the first step: **patches → tokens.** Once an image is a sequence of tokens, everything you learned about attention applies *unchanged*.

---

# Why throw away CNNs — the 2020 bet

<div class="paper">

Dosovitskiy et al. 2021 · *"An Image is Worth 16×16 Words"* — split the image into patches, treat them as tokens, apply a vanilla Transformer. Drop convolutions entirely.

</div>

<div class="keypoint">

CNNs have vision **baked in**: forced to look at local pixels first, then build up to objects (L9–L11). The bet: hand a generic Transformer the *whole* image at once and let it **learn** what matters.

</div>

With enough data, **fewer prior assumptions + more capacity** beats a hand-engineered architecture. By 2021 (ViT-Huge, pretrained on 300M images) the bet paid off: ViT **surpassed** the best ResNets on ImageNet.

---

# How can a Transformer "read" an image?

<div class="insight">

**Analogy · describing a photo over the phone.** You don't recite every pixel. You break it into chunks: *"top-left, blue sky · below it, a green tree…"*

We teach the model the same move: chop the image into a grid of **patches** — "image words" — and let attention decide which patches matter to which.

</div>

---

# ViT · the object, in a picture

![w:900px](figures/lec18/svg/vit_patches_tokens.svg)

Cut into patches → flatten + linearly project each → prepend a `[CLS]` token, add position embeddings → a sequence of tokens into a standard Transformer.

---

# From pixels to tokens · the recipe

A 4×4 grayscale image, 2×2 patches → $(4/2)^2 = 4$ patches.

$$\text{Image} = \begin{pmatrix} 10 & 20 & 150 & 160 \\ 30 & 40 & 170 & 180 \\ 190 & 200 & 5 & 15 \\ 210 & 220 & 25 & 35 \end{pmatrix}$$

1. **Patchify** — patch 1 (top-left) $=\begin{pmatrix}10 & 20\\30 & 40\end{pmatrix}$.
2. **Flatten** — patch 1 $\to [10,20,30,40]$.
3. **Linearly project** — multiply by a learned $W$ to get a $d$-dim embedding (the "token").
4. **Prepend `[CLS]`, add position embeddings** — sequence `[CLS, p1, p2, p3, p4]`.

---

# Worked numeric · one patch embedding

Flattened patch $x = [10, 20, 30, 40]$, learned $W$ (size $4\times 3$, output dim $d=3$):

$$W = \begin{pmatrix} 0.1 & 0 & -0.2 \\ 0 & 0.5 & 0 \\ -0.1 & 0 & 0.3 \\ 0.2 & -0.1 & 0 \end{pmatrix}$$

**Compute** $\;\text{embedding} = xW$:

- $10(0.1) + 30(-0.1) + 40(0.2) = 1 - 3 + 8 = 6.0$
- $20(0.5) + 40(-0.1) = 10 - 4 = 6.0$
- $10(-0.2) + 30(0.3) = -2 + 9 = 7.0$

$$\text{embedding} = \mathbf{[6.0,\ 6.0,\ 7.0]}$$

— the first token the Transformer sees, standing in for the top-left of the image.

---

# From pixels to tokens · real ViT numbers

For a real **224×224 RGB** image with **16×16** patches:

<div class="math-box">

- Each patch flattens to $16\cdot 16\cdot 3 = 768$ numbers.
- Patches per image: $(224/16)^2 = 14^2 = \mathbf{196}$.
- Project each to $d = 768$; prepend `[CLS]` → **197 tokens** into a standard Transformer.

</div>

The patchify step is *conceptually a strided convolution* with kernel = stride = patch size (appendix) — the only "vision-specific" op in the whole model.

---

# ViT · the full read-out

![w:900px](figures/lec18/svg/vit_patches.svg)

The `[CLS]` token summarizes the whole image → a linear head turns it into class logits.

<div class="notebook">

**🎛 Interactive · Vision Transformer** — drag over patches and watch which ones attention attends to; see the patch → token pipeline live. *(Interactive Lab · `interactive/articles/vision-transformer`)*

</div>

---

# ViT vs CNN · inductive biases

| | CNN | ViT |
|---|---|---|
| Receptive field | local → global, slowly | global from layer 1 |
| Weight sharing | spatial (same kernel everywhere) | none (attention per position) |
| Translation equivariance | baked in | learned, if at all |
| Small-data regime | strong | weak |
| Large-data regime | plateaus | keeps improving |

<div class="keypoint">

**CNNs encode vision priors; ViTs learn them.** With little data, the priors win. With massive data (300M+), learning them from scratch wins.

</div>

---

# ViT variants you'll meet

| Model | Patch | Params | Note |
|---|:-:|:-:|---|
| ViT-B/16 | 16×16 | 86M | "Base" — the default |
| ViT-L/14 | 14×14 | 307M | CLIP's image encoder |
| ViT-H/14 | 14×14 | 632M | SOTA around 2021 |
| DINOv2 ViT-g | 14×14 | 1.1B | 2023 general-purpose vision |

Smaller patches → more tokens → finer detail, but more compute (attention is quadratic in token count, L14). **Swin Transformer** re-introduces windows and hierarchy — a ViT–CNN hybrid that stays popular in practice.

---

# Practice problem 1

<div class="popquiz">

**Practice problem 1.** A **ViT-B/16** takes a $224\times 224$ RGB image with $16\times 16$ patches and embedding dimension $d = 768$.

(a) How many patch-tokens? How many tokens enter the Transformer?
(b) The patch embedding is one linear map from a flattened patch to $\mathbb R^{768}$. How many parameters (weights + bias) does that map have?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 1

**(a)** Patches: $(224/16)^2 = 14^2 = \mathbf{196}$. With the `[CLS]` token, $196 + 1 = \mathbf{197}$ tokens enter the Transformer.

**(b)** A flattened patch has $16\cdot 16\cdot 3 = 768$ inputs, mapped to $768$ outputs:

$$\underbrace{768 \times 768}_{\text{weights}} + \underbrace{768}_{\text{bias}} = 589{,}824 + 768 = \mathbf{590{,}592}\ \text{params}$$

<div class="insight">

Under 0.6M parameters turns pixels into tokens — a rounding error next to the 86M-parameter Transformer that follows. **The architecture is 99% "just a Transformer."**

</div>

---

<!-- _class: section-divider -->

## Part 2 · CLIP — a caption is a classifier

---

# CLIP in one sentence

<div class="keypoint">

**CLIP** has **two encoders** — one for images, one for text — that map into **one shared space**. Training pulls **matched image–caption pairs together** and pushes **mismatched pairs apart**, so afterwards *cosine similarity answers* "do this image and this caption match?"

</div>

This is **exactly L19's contrastive idea, across two modalities.** In L19 the two "views" were two crops of the *same image*; here the second view is the image's **caption**. Same pull-together / push-apart loss — image-vs-text instead of image-vs-image. Both encoders are Transformers: under the hood, still attention.

---

# CLIP · two encoders, one shared space

![w:900px](figures/lec18/svg/clip_dual_encoder.svg)

An image encoder (ViT) and a text encoder (Transformer) each emit a **unit vector**. Matched pairs should land close; everything else, far.

---

# CLIP · training as a contrast matrix

![w:900px](figures/lec18/svg/clip_training_matrix.svg)

One batch of $N$ pairs → an $N\times N$ similarity matrix. **Diagonal = correct pairs (pull up); off-diagonal = mismatches (push down).** This is InfoNCE (L19), extended across modalities.

---

# CLIP · from matrix to loss

For a batch of $N$ pairs, unit-normalize every embedding and form $S_{ij} = \text{image}_i \cdot \text{text}_j$:

1. **Scale by temperature** — $\text{logits} = S / \tau$.
2. **Row-wise cross-entropy** — each image should match its own caption (label $=$ diagonal).
3. **Column-wise cross-entropy** — each caption should match its own image.
4. **Average** — $\mathcal L = \tfrac12(\text{CE}_\text{rows} + \text{CE}_\text{cols})$.

<div class="math-box">

**By the numbers** · image encoder ViT-L/14 · text encoder 12-layer Transformer · **400M** web (image, caption) pairs · batch size **32,768** — so every step contrasts against tens of thousands of negatives at once.

</div>

---

# What the shared space buys you

Once matched pairs are close and mismatched far, the *same* space does three jobs — **no retraining**:

<div class="keypoint">

- **Retrieve** — find images from a text query (nearest caption embedding).
- **Classify zero-shot** — pick the closest *caption template* to the image.
- **Guide generation** — diffusion models steer on CLIP text features.

</div>

The next few slides unpack **zero-shot classification** — the move that made CLIP famous.

---

# Practice problem 2

<div class="popquiz">

**Practice problem 2.** A tiny CLIP batch, $N=2$. Unit embeddings: images $i_1=[1,0]$, $i_2=[0,1]$; captions $t_1=[0.8,0.6]$, $t_2=[0.6,0.8]$.

(a) Build the $2\times2$ cosine-similarity matrix $S_{ij}=i_i\cdot t_j$.
(b) With temperature $\tau = 0.2$, row-softmax **image 1**. Which caption does it match — and what are the *correct* matched pairs?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 2

**(a)** $S = \begin{pmatrix} i_1\cdot t_1 & i_1\cdot t_2 \\ i_2\cdot t_1 & i_2\cdot t_2 \end{pmatrix} = \begin{pmatrix} 0.8 & 0.6 \\ 0.6 & 0.8 \end{pmatrix}$

**(b)** Row 1 logits $= [0.8, 0.6]/0.2 = [4, 3]$.

$$\text{softmax}([4,3]) = \frac{[e^4, e^3]}{e^4+e^3} = \frac{[54.6,\ 20.1]}{74.7} \approx [\mathbf{0.73},\ 0.27]$$

<div class="insight">

Image 1 matches **caption $t_1$**; by symmetry image 2 matches $t_2$. The **diagonal** $(i_1,t_1),(i_2,t_2)$ is the set of correct pairs — exactly what the contrastive loss pushes toward. Notice $\tau$ sharpens a $0.8$-vs-$0.6$ gap into $0.73$-vs-$0.27$.

</div>

---

# Zero-shot classification · a caption is a classifier

CLIP has **no dog logit, no softmax head, no dog classifier** anywhere. Yet it labels a dog photo "dog." How?

<div class="keypoint">

Write each class as a caption — `"a photo of a dog"`, `"a photo of a cat"` — encode them **once**, and treat those caption vectors as **classifier weights**. Each new image is one encoder call plus a few dot products; **nearest caption wins.**

</div>

The text encoder *writes the classifier on the fly.* CLIP never saw ImageNet labels, yet beats a supervised ResNet-50 on zero-shot ImageNet.

---

# Practice problem 3

<div class="popquiz">

**Practice problem 3.** Toy CLIP space, $d=2$. An image embeds to $i = [0.6,\ 0.8]$ (unit). Three cached caption embeddings:

`"a photo of a cat"` $= [1,0]$ · `"a photo of a dog"` $= [0.6,\ 0.8]$ · `"a photo of a car"` $= [0,1]$.

Which class does CLIP predict, and *why* — in the language of "closest caption embedding"?

</div>

*Try it before the next slide.*

---

# Solution · practice problem 3

Cosine similarity (dot product of unit vectors):

$$i\cdot\text{cat} = 0.6, \qquad i\cdot\text{dog} = 0.36 + 0.64 = \mathbf{1.0}, \qquad i\cdot\text{car} = 0.8$$

<div class="insight">

The **dog** caption embedding is *identical* to the image embedding (cosine $=1$), so it is the closest — CLIP predicts **"dog."** Classification became a nearest-neighbour search in the shared space: **the class you can *name* closest is the class you predict.** No head, no fine-tuning, ever.

</div>

---

# The text encoder is free at inference

<div class="insight">

Encode your $K$ class captions **once** → a fixed $K\times d$ matrix. It never re-runs per image: it has literally *become* a classifier's weight matrix. Each new image is then **one** image-encoder pass plus $K$ cheap dot products.

</div>

100 classes? Encode 100 captions once, reuse forever. Want new classes? *Type new captions* — no gradients, no data, no retraining. That is what "a caption is a classifier" buys you in compute.

---

# Why CLIP mattered

Zero-shot accuracy is sensitive to the **caption template** — prompt engineering, but for vision:

| Template | Zero-shot ImageNet |
|:--|:-:|
| `"{label}"` (bare word) | 63% |
| `"a photo of a {label}"` | 68% |
| `"a photo of a {label}, a type of pet"` (pets) | 72% |

<div class="notebook">

**🎛 Interactive · CLIP zero-shot** — drop in an image, edit the candidate captions, watch the softmax move. *(Interactive Lab · `interactive/articles/clip-zero-shot`)* · **📓 Notebook** — load pretrained CLIP, classify custom images, sweep prompts. *(ES 667 · `notebooks/20-clip-zero-shot.ipynb`)*

</div>

Matching the prompt to the training caption distribution *is* the trick — and CLIP text features now underpin Stable Diffusion, DALL·E 2, and every VLM that follows.

---

<!-- _class: section-divider -->

## Part 3 · LLaVA — vision is just more tokens

---

# LLaVA in one sentence

<div class="keypoint">

A small **projection layer** maps CLIP's image tokens into the **LLM's token space**, so the LLM reads image tokens *exactly like word tokens* — and answers in text. That one bridge turns a text-only LLM into a model that can **see and talk**.

</div>

The pieces already exist: **CLIP/ViT** gives image tokens (Parts 1–2), the **LLM** (L15–L17) generates text. LLaVA only learns the **glue** between them.

---

# LLaVA · the translator analogy

<div class="insight">

Two experts speak different languages.
- **CLIP** understands images — vectors in $\mathbb R^{1024}$.
- **Llama** understands text — vectors in $\mathbb R^{4096}$.

We need a **translator**: a linear map $W$ that rewrites an image vector as something Llama can read. Both spaces already encode similar concepts (*puppy* near *dog*), so the translator mostly has to **rotate and scale**.

</div>

---

# LLaVA · the full stack

![w:900px](figures/lec18/svg/llava_stack.svg)

CLIP (frozen) → **linear projection** → image tokens sit right next to the text tokens → the LLM decodes an answer.

---

# LLaVA · the projection, with shapes

1. **CLIP output** — $256$ patch embeddings, each $1024$-d → tensor `[256, 1024]`.
2. **LLM expects** — token embeddings of dim $4096$.
3. **Bridge** — a linear layer $W$ of shape `[1024, 4096]` with bias `[4096]`.

Per patch: `llm_token = patch @ W + b`, shape check `[1,1024] @ [1024,4096] → [1,4096]` ✓.

Do this for all 256 patches, then **prepend** them to the user's text tokens:

$$[\text{img}_1, \ldots, \text{img}_{256},\ \text{text}_1, \text{text}_2, \ldots] \to \text{LLM}$$

---

# Worked numeric · the projection

CLIP patch $\text{patch} = [2.0,\ -1.0,\ 0.5]$ (3-d for clarity); the LLM expects 4-d tokens.

$$W = \begin{pmatrix} 1 & 0 & 0.5 & 0.1 \\ 0 & 2 & 0 & -0.2 \\ 0.1 & -1 & 0 & 0 \end{pmatrix},\quad b = [0,\ 0,\ 0.1,\ 0]$$

**Compute** $\text{patch}\, W + b$:

- $2(1) + (-1)(0) + 0.5(0.1) = 2.05$
- $2(0) + (-1)(2) + 0.5(-1) = -2.5$
- $2(0.5) + 0 + 0 \;+\; 0.1 = 1.1$
- $2(0.1) + (-1)(-0.2) + 0 = 0.4$

$$\text{llm\_token} = \mathbf{[2.05,\ -2.5,\ 1.1,\ 0.4]}$$

The LLM treats this vector exactly like the embedding for the word "cat."

---

# LLaVA · the recipe & two-stage training

<div class="math-box">

**LLaVA (Liu et al. 2023):** CLIP ViT-L extracts image features → a **projection** maps them into the LLM's embedding space (LLaVA-1: one linear layer; LLaVA-1.5: a 2-layer MLP) → concatenate `[image, text]` → LLM.

</div>

<div class="columns">
<div>

**Stage 1 · align**
Freeze CLIP *and* the LLM; train **only** $W, b$ on image–caption data. The projection learns the map into word space.

</div>
<div>

**Stage 2 · instruction-tune**
Unfreeze the LLM; fine-tune on `(image, question, answer)` triples so it learns to *converse* about images.

</div>
</div>

New params: ~4M for the linear bridge — **0.06%** of a 7B model. Now it clicks: **vision became just more tokens.**

---

# See a VLM answer, end to end

<div class="notebook">

**🎛 Interactive · VLM** — feed an image + a question; watch CLIP features get projected and the LLM decode an answer token by token; toggle the projection off to see it break. *(Interactive Lab · `interactive/articles/vlm`)*

</div>

<div class="notebook">

**📓 Notebook · VLM** — run a LLaVA-style stack: CLIP encode → linear projection → prepend to a text prompt → generate. Inspect the 256 image tokens as ordinary vectors. *(ES 667 · VLM notebook)*

</div>

<div class="popquiz">

**Discuss.** In stage 1 only $W, b$ train while CLIP *and* the LLM stay frozen. Why is aligning that one linear map already enough to make the model "see"?

</div>

---

# An alternative bridge · Flamingo's cross-attention

<div class="columns">
<div>

**LLaVA** — project image tokens *into* the LLM's input sequence; the LLM reads them as ordinary tokens. Simple, and dominant in open source.

</div>
<div>

**Flamingo** (Alayrac 2022) — keep the sequence text-only; insert **cross-attention** layers that *query* image features when needed, via a learned Perceiver Resampler.

</div>
</div>

<div class="insight">

Same goal, two couplings: **concatenate** (LLaVA) versus **cross-attend** (Flamingo). Both are, once more, just attention (L14) wired differently — LLaVA won open source on sheer simplicity.

</div>

---

<!-- _class: section-divider -->

## Part 4 · The frontier, briefly

---

# Native-multimodal vs bolt-on

<div class="columns">
<div>

### Bolt-on (LLaVA, Flamingo)
- Take a finished LLM, hire a translator (projection).
- Vision tower stays "foreign" to the LLM; weaker on tight text–vision interaction.
- **Cheap** — only train the bridge. Dominant in open source.

</div>
<div>

### Native (Gemini, GPT-4o, Claude)
- Pretrain on interleaved text + image + audio from the start.
- Unified tokenization → stronger multimodal reasoning.
- **Expensive** — pretrain a huge model from zero.

</div>
</div>

<div class="insight">

The input side is increasingly "anything"; the output is still mostly **text**. True any-to-any (text ↔ image ↔ video ↔ audio) is the current frontier.

</div>

---

# Hallucination · priors fighting evidence

<div class="keypoint">

The LLM has read "a cup of coffee on a desk" **millions of times** in text. Hand it a *blurry* image of a desk, and the language **prior** can override the weak visual signal — the model adds a coffee cup that isn't there.

</div>

It's a tug-of-war between **prior** (text statistics) and **evidence** (this image). Strong evidence (a clear photo) wins easily; weak evidence lets the prior "fill in" confident fictions. Fixes: stronger vision encoders, more VQA data, and RLHF targeted at vision — real progress, not yet solved.

---

<!-- _class: summary-slide -->

# One sentence to remember

<div class="keypoint">

**Cut an image into patches, embed them as tokens, and it's all attention: ViT makes an image a sequence, CLIP puts image and text in one shared space (a caption is a classifier), and LLaVA lets an LLM read image tokens like words.**

</div>

- **ViT** — patches → tokens → a plain Transformer; no convolutions.
- **CLIP** — dual encoder, contrastive (L19 across modalities); zero-shot = nearest caption.
- **LLaVA** — a projection layer drops image tokens into the LLM's token space.
- **Frontier** — native vs bolt-on; hallucination is priors overriding weak evidence.

**Next:** generation — encode inputs to *distributions*, not points (VAEs), then diffusion.

---

<!-- _class: compact -->

# Sources & credits

<div class="paper">

This lecture reuses and adapts material from the instructor's ES 667 course (IIT Gandhinagar) and standard references:

- **ViT** — Dosovitskiy et al., *"An Image is Worth 16×16 Words,"* ICLR 2021.
- **CLIP** — Radford et al., *"Learning Transferable Visual Models from Natural Language Supervision,"* 2021.
- **LLaVA** — Liu et al., *"Visual Instruction Tuning,"* NeurIPS 2023 (LLaVA-1.5, 2023).
- **VLM teaching materials, notebook & interactives** — ES 667 Deep Learning, N. Batra · Interactive Lab: `interactive/articles/{vision-transformer, clip-zero-shot, vlm}`.
- **Pedagogical framing** — A. Ng, *Deep Learning Specialization* (build ideas up from pieces; introduce concepts in-line).

Figures adapted from the ES 667 figure library (`figures/lec18/svg/`). All source material © N. Batra & teaching staff.

</div>

---

<!-- _class: section-divider -->

## ⭐⭐⭐ Appendix · optional depth

*Skip on a first pass — for the curious.*

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · CLIP's symmetric InfoNCE loss

For a batch of $N$ pairs with unit embeddings and $s_{ij} = \text{image}_i \cdot \text{text}_j$, temperature $\tau$:

$$\mathcal L = -\frac{1}{2N}\sum_{i=1}^{N}\Bigg[\underbrace{\log\frac{e^{s_{ii}/\tau}}{\sum_{j} e^{s_{ij}/\tau}}}_{\text{image }i\to\text{ its caption}} \;+\; \underbrace{\log\frac{e^{s_{ii}/\tau}}{\sum_{j} e^{s_{ji}/\tau}}}_{\text{caption }i\to\text{ its image}}\Bigg]$$

This is precisely L19's InfoNCE with the positive being the diagonal pair and all $N-1$ off-diagonal entries as negatives — applied **symmetrically** over rows (images) and columns (texts). $\tau$ is *learned* (stored as $\log(1/\tau)$ for stability). Larger batches ⇒ more negatives ⇒ a sharper space, which is why CLIP trains at batch 32,768.

---

<!-- _class: math-heavy -->

# ⭐⭐⭐ Optional · patch embedding = a strided convolution

Patchify + linear-project is *identical* to one convolution with kernel = stride = patch size:

$$\texttt{nn.Conv2d(3, d, kernel\_size=16, stride=16)}$$

Non-overlapping $16\times16$ receptive fields, each producing one $d$-vector — exactly the 196 patch tokens. Its parameter count:

$$\underbrace{3\cdot 16\cdot 16 \cdot d}_{\text{weights}} + \underbrace{d}_{\text{bias}} = 768d + d \;\xrightarrow{d=768}\; 590{,}592$$

Add the learned position embeddings $(196{+}1)\times 768 = 151{,}296$ and the `[CLS]` token $768$ → the full embedding front-end is $\approx 742{,}656$ params. Everything after is a vanilla Transformer — the single conv is the *only* vision-specific operation in ViT.
