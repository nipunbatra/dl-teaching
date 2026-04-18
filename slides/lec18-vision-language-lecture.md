---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Vision-Language Models

## Lecture 18 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Where we are

- **CNNs** (L7–L9) · vision-specific inductive bias.
- **Transformers** (L13–L14) · originally for text.
- **Self-supervised** (L17) · contrastive and masked pretraining.

Today: what if one model handled **both** text AND vision?

<div class="paper">

Today maps to **Prince Ch 12 §12.5** (ViT) + the CLIP, LLaVA, Flamingo papers. This is where the full "multimodal" LLM came from.

</div>

Four questions:
1. How do Transformers process images (no convolutions)?
2. What did **CLIP** unlock?
3. How does **LLaVA** give an LLM eyes?
4. What's the 2026 multimodal state?

---

<!-- _class: section-divider -->

### PART 1

# Vision Transformer (ViT)

Apply Transformer to images — no convolutions

---

# The 2020 bet

<div class="paper">

Dosovitskiy et al. 2020 · *"An Image is Worth 16×16 Words"* — split image into patches, treat them as tokens, apply vanilla Transformer. Drop convolutions entirely.

</div>

Controversial at the time — CNNs had reigned for 8 years. The bet: if you have enough data and compute, the right architecture is the one with the fewest inductive biases.

It worked. ViT-Huge pretrained on 300M images beat CNN SOTA on ImageNet by 2021.

---

# How ViT works

![w:920px](figures/lec18/svg/vit_patches.svg)

---

# ViT vs CNN · inductive biases

| | CNN | ViT |
|---|-----|-----|
| Receptive field growth | local → global slowly | global from layer 1 |
| Weight sharing | spatial | none (each position has own attention) |
| Translation equivariance | baked in | learned (if at all) |
| Data efficiency (small data) | strong | weak |
| Data efficiency (large data) | plateaus | keeps improving |

<div class="keypoint">

**CNNs encode vision priors; ViTs learn them.** With small data, priors win. With massive data (300M+), learning wins.

</div>

---

# ViT variants you'll encounter

| Model | Patch | Params | Notes |
|-------|-------|--------|-------|
| ViT-B/16 | 16×16 | 86M | "Base" — most common |
| ViT-L/14 | 14×14 | 307M | Used by CLIP |
| ViT-H/14 | 14×14 | 632M | SOTA around 2021 |
| ViT-g (DINOv2) | 14×14 | 1.1B | 2023 general-purpose vision |

**Swin Transformer** (Liu et al. 2021) adds hierarchical windowing — bridges ViT and CNN. Popular in practice.

---

<!-- _class: section-divider -->

### PART 2

# CLIP · contrastive image-text pretraining

The paper that launched zero-shot vision

---

# CLIP · dual encoder

![w:920px](figures/lec18/svg/clip_dual_encoder.svg)

---

# CLIP · training setup

<div class="math-box">

**Contrastive loss** (identical to SimCLR but across modalities):

$$\mathcal{L} = \frac{1}{2}\,[\,\text{CE}_\text{rows}(\text{logits}) + \text{CE}_\text{cols}(\text{logits})\,]$$

where $\text{logits}_{ij} = I_i \cdot T_j / \tau$ and the target for row $i$ is column $i$.

</div>

- **Image encoder** — ViT-L/14 (or ResNet-50 for small variant).
- **Text encoder** — a 12-layer Transformer.
- **Training data** — 400M image-text pairs scraped from the web (OpenAI private).
- **Batch size** — 32,768 (big!).

---

# CLIP's killer feature · zero-shot

```python
import clip
model, preprocess = clip.load("ViT-L/14")

image = preprocess(pil_image).unsqueeze(0)
texts = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
text_tokens = clip.tokenize(texts)

with torch.no_grad():
    img_emb  = model.encode_image(image)
    txt_embs = model.encode_text(text_tokens)

# Cosine similarity → pick the closest text
sims = (img_emb @ txt_embs.T).softmax(dim=-1)
# sims = [[0.89, 0.08, 0.03]]  → "cat"
```

**No training on cats.** CLIP has never seen an ImageNet label. Yet it beats ResNet-50 on zero-shot ImageNet classification.

---

# Why CLIP mattered

1. **Universal image representation** — one encoder works on any domain.
2. **Prompt engineering for vision** — "a photo of X" vs "a sketch of X" vs "a satellite image of X" shifts behavior without retraining.
3. **Foundation for everything that came next** — Stable Diffusion uses CLIP's text encoder; DALL-E 2, Flamingo, LLaVA all build on CLIP features.

<div class="realworld">

CLIP is the **default general-purpose vision-language model** in 2026. For retrieval, search, zero-shot classification, content moderation, CLIP features are the first thing you try.

</div>

---

<!-- _class: section-divider -->

### PART 3

# LLaVA · give an LLM eyes

Project image features into token space

---

# LLaVA · the architecture

```
image → ViT-L/14 (CLIP encoder) → 256 image tokens (frozen)
                                              ↓
                                     Linear projection
                                              ↓
                                       "Image tokens" in LLM space
                                              ↓
                          Llama 2 / Vicuna (autoregressive)
                                              ↓
                                       text response
```

<div class="math-box">

**LLaVA recipe** (Liu et al. 2023):

1. Pretrained CLIP ViT-L extracts 256 image features.
2. Simple MLP projects them into the LLM's token embedding space.
3. Concatenate `[image_tokens, text_tokens]` and feed to an LLM.
4. Fine-tune with instruction data: `(image, question, answer)` triples.

</div>

Surprisingly good. The LLM brings reasoning; CLIP brings vision understanding; the projection layer glues them.

---

# Flamingo · cross-attention bridge

Alayrac et al. 2022 · an alternative approach:

- Keep the LLM frozen.
- Insert **cross-attention** layers between LLM blocks that attend to image features.
- Use a **Perceiver Resampler** — a learned set of query tokens that distill image features.

Benefit: the LLM never "sees" images as tokens; it queries them via cross-attention when it needs to.

<div class="insight">

Both approaches work. LLaVA is simpler and became dominant in open-source; Flamingo-style crossattention persists in frontier labs (Anthropic's vision, Google's Gemini variants).

</div>

---

<!-- _class: section-divider -->

### PART 4

# Multimodal LLMs in 2026

The frontier

---

# 2026 multimodal state

| Model | What it sees | What it does |
|-------|--------------|--------------|
| **GPT-4V / GPT-5** | image, video | general reasoning + tool use |
| **Claude 4 (Anthropic)** | image, PDF, video frames | general reasoning + computer use |
| **Gemini 2 Ultra (Google)** | image, audio, video | natively multimodal from pretraining |
| **LLaVA / Qwen-VL (open)** | image | open-source equivalent of GPT-4V |

**Key trend**: the input side is increasingly "anything", the output is still mostly text. True any-to-any (text→image→video→audio) is the 2026+ frontier.

---

# Practical multimodal prompting

```python
from anthropic import Anthropic
client = Anthropic()

response = client.messages.create(
    model="claude-4",
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "url", "url": "https://..."}},
                {"type": "text", "text": "What does this plot suggest?"}
            ]
        }
    ]
)
```

Three lines in; a paragraph of reasoning out. The API abstracts away all the CLIP + projection + LLM machinery.

---

# Emerging applications

<div class="columns">
<div>

### Vision tasks, zero-shot
- Segment Anything prompts
- OCR without explicit training
- Diagram understanding
- Chart reading

</div>
<div>

### Agentic loops
- **Computer use** — click buttons from screenshots
- **Robotic manipulation** — VLM → action tokens
- **Content moderation**
- **Code + diagram refactoring**

</div>
</div>

The agentic side (Claude computer use, GPT operator) is where multimodal is most valuable in 2026.

---

<!-- _class: summary-slide -->

# Lecture 18 — summary

- **ViT** · patches → tokens → Transformer. No convolutions. Scales with data; CNNs plateau.
- **CLIP** · dual encoder, contrastive on 400M image-text pairs · zero-shot vision via text prompts.
- **LLaVA** · CLIP features → linear projection → LLM token space. Simple, effective.
- **Flamingo** · cross-attention bridge with Perceiver Resampler. Alternate approach.
- **2026** · every frontier LLM is multimodal (GPT-5, Claude 4, Gemini 2). Output side still text-dominated.

### Read before Lecture 19

Prince Ch 17 · Variational Autoencoders.

### Next lecture

**Autoencoders &amp; VAEs** — compression, latent spaces, the reparameterization trick, ELBO.

<div class="notebook">

**Notebook 18** · `18-clip-zero-shot.ipynb` — load pretrained CLIP; zero-shot classify custom images; sweep text prompts to see how wording changes the result.

</div>
