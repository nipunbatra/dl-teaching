---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Diffusion Models — Practice

## Lecture 22 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. Condition a diffusion model on **text** via cross-attention.
2. Derive the **classifier-free guidance** update rule and pick $w$.
3. Explain why **latent diffusion** made Stable Diffusion shippable.
4. Describe **DDIM** sampling and skip 95% of steps with no retraining.
5. Recognize **DiT** (diffusion Transformer) as U-Net's replacement.
6. Place **flow matching** and **consistency models** as the 2026 frontier.

---

# Where we are

Last lecture · **DDPM** — forward noise, learn reverse, predict ε. Works on MNIST, toy 2D. But how do we get from there to **Stable Diffusion** and **Sora**?

<div class="paper">

Today maps to **Prince Ch 18 (later sections)** + HF `diffusers` docs + Rombach 2022 (Stable Diffusion) + Ho &amp; Salimans 2022 (CFG).

</div>

Four questions:

1. How do we **condition** on a text prompt?
2. What is **classifier-free guidance**?
3. Why does **latent diffusion** matter?
4. What about **faster** sampling — DDIM and DiT?

---

<!-- _class: section-divider -->

### PART 1

# Conditioning · from unconditional to text-to-image

---

# How to condition a diffusion model

We want $\epsilon_\theta(x_t, t, c)$ — predict noise given a **condition** $c$ (class label, text embedding, image).

Three common ways to inject $c$:

1. **Concatenate** $c$ to the input channels (for class labels).
2. **Add** $c$ to the time embedding (simple but loses detail).
3. **Cross-attention** between $c$ and intermediate features (Stable Diffusion, most modern).

---

# Cross-attention · the painter-following-instructions analogy

<div class="keypoint">

How does an image model **listen** to a text prompt?

Imagine a painter following instructions. For every brushstroke they ask, *"Which part of the instruction applies right here?"*

Cross-attention is exactly this · each spatial location in the image listens to the most relevant words in the prompt.

</div>

The result · a dynamic, **spatial** link between text and pixels. The "cat" word strongly influences cat-shaped regions; the "background" word influences corners.

---

# Text conditioning in Stable Diffusion

```python
# 1. Encode the prompt
text_emb = clip_text_encoder("a cat astronaut")  # [1, 77, 768]

# 2. Inside the U-Net, every attention block has cross-attention to text_emb
class CrossAttention(nn.Module):
    def forward(self, spatial_features, text_emb):
        Q = self.q_proj(spatial_features)        # from image features
        K = self.k_proj(text_emb)                # from text embedding
        V = self.v_proj(text_emb)
        return softmax(Q @ K.T / sqrt(d)) @ V
```

At each diffusion step, every spatial position can **attend to** any text token — the model learns to line up image content with prompt words.

---

# Cross-attention · why it works for text conditioning

Self-attention inside the U-Net mixes spatial positions. **Cross**-attention *between spatial features and text features* is where the prompt enters.

<div class="keypoint">

At a high-resolution block · each pixel asks "which prompt tokens matter for me?". A cat pixel pays attention to "cat" in the prompt; a sky pixel to "sky". The result is **spatially localized conditioning** — a single prompt can steer different regions differently.

</div>

Visualizing the cross-attention maps reveals exactly this — each word has a blob of pixels that listened most to it. This is the mechanism behind DreamBooth, Prompt-to-Prompt, and every prompt-editing technique.

---

<!-- _class: section-divider -->

### PART 2

# Classifier-Free Guidance

The trick that makes generation feel "on-prompt"

---

# CFG · the geometry

![w:920px](figures/lec22/svg/cfg_guidance_vectors.svg)

---

# CFG · the extrapolation trick

**Intuition** · at each denoising step, the unconditional prediction is "what *any* image wants to do right now." The conditional prediction is "what a *prompt-matching* image wants to do." The **difference vector** points *toward* the prompt.

<div class="math-box">

$$\epsilon_\text{CFG} = \epsilon_\theta(x_t, \emptyset) + w \cdot \big(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\big)$$

- $\epsilon_\theta(x_t, \emptyset)$ · unconditional noise prediction (null prompt).
- $\epsilon_\theta(x_t, c)$ · conditional, given prompt $c$.
- $w$ · the guidance scale. Default ~7.

Take that difference and **walk $w\times$ as far in that direction**. $w = 1$ gives plain conditional; $w > 1$ over-shoots to amplify prompt adherence; $w = 0$ ignores the prompt.

</div>

---

# CFG in pictures

![w:920px](figures/lec22/svg/cfg_scale.svg)

<div class="realworld">

▶ Interactive: slide $w$ from 0 to 30, watch prompt adherence vs artifacts — [cfg-scale-visualizer](https://nipunbatra.github.io/interactive-articles/cfg-scale-visualizer/).

</div>

---

# Training with CFG-ready dropout

To enable CFG at inference, the training needs both conditional and unconditional examples:

```python
def training_step(x0, prompt):
    t = sample_t()
    noise = torch.randn_like(x0)
    x_t = add_noise(x0, noise, t)

    # Dropout 10% of prompts to null (enables unconditional path later)
    if random.random() < 0.10:
        c = null_embedding
    else:
        c = clip_text_encoder(prompt)

    pred_noise = model(x_t, t, c)
    return F.mse_loss(pred_noise, noise)
```

Same network learns both modes. At inference, you run it twice and extrapolate. Cost · 2× compute per step. Benefit · any w at generation time.

---

# CFG · vectors in 2D

![w:920px](figures/lec22/svg/cfg_arithmetic.svg)

---

# Worked example · CFG arithmetic at one step

<div class="math-box">

At denoising step $t$, the U-Net runs twice on the noisy latent $x_t$:

- $\epsilon_\emptyset = \epsilon_\theta(x_t, \text{null prompt}) = [+0.10, -0.30]$ (toy 2D)
- $\epsilon_c = \epsilon_\theta(x_t, \text{"cat"}) = [+0.40, -0.10]$

Difference vector · $\Delta = \epsilon_c - \epsilon_\emptyset = [+0.30, +0.20]$.

| $w$ | $\epsilon_\text{cfg} = \epsilon_\emptyset + w \Delta$ | meaning |
|:-:|:-:|:-:|
| 0 | $[+0.10, -0.30]$ | unconditional |
| 1 | $[+0.40, -0.10]$ | pure conditional |
| 3 | $[+1.00, +0.30]$ | overshoot 2× |
| 7 | $[+2.20, +1.10]$ | strong overshoot |

</div>

Larger $w$ pushes the noise prediction further along the "more cat-like" direction. Subtract that bigger noise → step lands further toward a strongly cat-shaped image.

---

# Picking a CFG scale · practical guide

| $w$ | What you get |
|:-:|:-:|
| 1 | pure conditional · ignores extrapolation |
| 3 | subtle prompt adherence · natural-looking |
| 7 | Stable Diffusion default · balanced |
| 12+ | strong adherence · saturated colors, artifacts |
| 25+ | cartoonish oversaturation; often broken |

<div class="insight">

CFG trades **diversity for prompt adherence**. Low $w$ produces many different interpretations; high $w$ produces a narrower, more on-prompt distribution. Treat $w$ as a hyperparameter you sweep for each task.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Latent diffusion

The one trick that made Stable Diffusion ship

---

# Why diffuse in latent space · the intuition

Most pixels in an image are *correlated*. A patch of blue sky has hundreds of near-identical pixel values. Diffusing each one independently is wasteful.

<div class="keypoint">

A pretrained VAE has already absorbed the **perceptually redundant** structure · the latent is a near-minimal representation. Diffusion then only has to model the *interesting* (high-entropy) dimensions.

</div>

Result · 48× fewer dimensions, same perceptual quality, ~10× faster sampling. This is the single change that made Stable Diffusion runnable on consumer GPUs.

---

# The problem with pixel-space diffusion

512×512×3 = **786,432 dimensions** per image. Running a 1B-param U-Net for 50 steps at that resolution is expensive — even on a single GPU this takes tens of seconds to minutes.

<div class="keypoint">

**Rombach et al. 2022 idea** — don't diffuse in pixel space. Use a **VAE** to compress images to a small latent space first, then diffuse there.

</div>

512×512×3 image → 64×64×4 latent → **48× fewer dimensions.**

Diffuse in latent space; decode once at the end.

---

# Stable Diffusion architecture

![w:920px](figures/lec22/svg/latent_diffusion.svg)

---

# Stable Diffusion · annotated full stack

![w:920px](figures/lec22/svg/latent_diffusion_stack.svg)

---

# Latent diffusion · three components

<div class="columns">
<div>

### 1. VAE (frozen)

Encoder: image → 4-channel latent.
Decoder: latent → image.
Pretrained, not updated during diffusion training.

### 2. CLIP text encoder (frozen)

Tokenize → embed → 77 × 768 tensor.
Frozen — the diffusion model just consumes these.

</div>
<div>

### 3. U-Net diffuser (trainable)

Operates in the 64×64×4 latent space.
Cross-attention injects text embeddings at every scale.
~1B parameters for SD v1.5.

This is the only thing you train.

</div>
</div>

---

<!-- _class: section-divider -->

### PART 4

# Faster sampling

DDIM and DiT

---

# Why 1000 steps · the quick math

DDPM's forward process adds tiny Gaussian noise at each of $T$ steps. To keep the final distribution close to $\mathcal{N}(0, I)$ and each step's noise increment small (so the reverse can be Gaussian too), $T$ has to be large — 1000 is the typical choice.

<div class="keypoint">

Small per-step noise → easier inverse problem for the network. But then the reverse loop has **1000 forward passes** — painful at inference.

</div>

DDIM (next slide) breaks this by reinterpreting the reverse process as *non-Markovian*, so you can skip steps without retraining.

---

# DDPM is slow · 1000 steps

Vanilla DDPM sampling requires running the U-Net **1000 times per generation**. Each step is a forward pass of a ~1B param network.

We want to make it faster without retraining. Two approaches:

- **Fewer sampling steps** (DDIM, DPM-Solver)
- **Better architecture** (DiT, replacing U-Net with Transformer)

---

# DDIM · the foggy-hill analogy

<div class="keypoint">

Original DDPM is like walking down a rocky hill in **fog**. You take 1000 tiny careful steps because you can only see one step ahead.

DDIM realizes · with the right model, **you can see the whole path**. Instead of 1000 wobbly steps, take 50 confident strides directly toward the final image.

</div>

The same trained model gets used · DDIM just chooses which subset of timesteps to evaluate. No retraining. 20× speedup at no cost in sample quality.

---

# DDIM · skip 95% of steps

![w:920px](figures/lec22/svg/ddim_vs_ddpm.svg)

---

# DDIM · deterministic sampling in 20–50 steps

<div class="paper">

Song, Meng, Ermon 2020 · *"Denoising Diffusion Implicit Models"*

</div>

DDIM reformulates the reverse process to be **deterministic** given the initial noise. Crucially, the trained DDPM model can be sampled with DDIM — no retraining.

**Effect** · 50 DDIM steps ≈ 1000 DDPM steps in quality. **20× speedup** essentially for free.

In 2026, DDIM (and its successor DPM-Solver++) is the default sampler in every diffusion library.

---

# DiT · replace U-Net with Transformer

<div class="paper">

Peebles &amp; Xie 2023 · *"Scalable Diffusion Models with Transformers"*

</div>

Same idea as ViT · patchify the latent, process with a pure Transformer. Advantages:

- **Better scaling** — Transformers eat more compute gracefully.
- **Simpler architecture** — no hand-designed U-Net ladder.
- **Shared toolchain** — the LLM ecosystem's optimizations (FlashAttention, tensor parallelism) carry over.

2024+ · DiT is the backbone of **Sora**, **Stable Diffusion 3**, and most new high-quality image/video models.

---

<!-- _class: section-divider -->

### PART 5

# The generative landscape · 2026

---

# What ships in 2026

| System | Architecture | Notes |
|--------|-------------|-------|
| **Stable Diffusion 3** | DiT + rectified flow | open-weight, commercial |
| **DALL-E 3** | diffusion (details undisclosed) | integrated with ChatGPT |
| **Midjourney v6+** | custom diffusion | paid, high fidelity |
| **Sora** | DiT on spacetime patches | OpenAI video model |
| **Veo 2** | latent diffusion, video | Google |
| **Flux** | open-weight SDXL successor | commercial, fast |

All diffusion-based. All descendants of the 2020 DDPM paper.

---

# Diffusion for non-image modalities

- **Text** · text-diffusion (still exploratory; LLMs beat it for quality).
- **Audio** · AudioGen, Riffusion, MusicLM.
- **3D** · DreamFusion, Gaussian Splatting integrates with diffusion priors.
- **Molecules** · RFdiffusion designs proteins (Baker lab, won 2024 Nobel adjacent).
- **Robotics** · Diffusion Policy (Chi 2023) — action generation via diffusion.

Diffusion is one of the two big generative paradigms now (the other: autoregressive LLMs).

---

# 2026 · the sampler menu

<div class="math-box">

| Sampler | Steps | Quality | Notes |
|:-:|:-:|:-:|:-:|
| DDPM (original) | 1000 | baseline | slowest, stochastic |
| DDIM (deterministic) | 50-100 | = | same model; drop-in |
| DPM-Solver++ | 20-30 | = | ODE solver · default in HF diffusers |
| Flow matching ODE | 8-20 | = | straighter trajectories |
| Consistency models | **1-4** | slight drop | distilled; near-real-time |

</div>

<div class="insight">

5-year trajectory · 1000 steps (2020) → 50 steps (DDIM) → 4 steps (consistency) → 1 step (Rectified Flow v3 distilled). Each generation reduced inference cost by ~10×.

</div>

---

# DiT · Transformer backbone

Peebles &amp; Xie 2022 · replace U-Net entirely with a Transformer.

<div class="math-box">

- **Patchify** the noisy latent (like ViT).
- **Transformer blocks** with attention + FFN.
- **Time + class conditioning** via adaLN (adaptive LayerNorm).
- Scales cleanly · more params → better (U-Net plateaus).

</div>

<div class="realworld">

Stable Diffusion 3 and Sora both use DiT-based architectures. DiT inherits Transformer's scaling laws · more data + compute = better samples, indefinitely.

</div>

---

# Diffusion · by modality in 2026

<div class="math-box">

| Modality | Model | Pixels / sec to generate |
|:-:|:-:|:-:|
| Images 1024² | SD3 · Flux | ~0.5 images/s (20 steps) |
| Video 720p · 16s | Sora · VEO | minutes per clip |
| Audio music | AudioGen · MusicLM | real-time (8 DDIM steps) |
| 3D meshes | Diffusion-SDF · ShapE | ~30s per mesh |
| Molecules | RFdiffusion | 10s per protein structure |
| Robot policies | Diffusion Policy | 50 Hz control loop |

</div>

<div class="insight">

The diffusion paradigm scaled to every signal that can be noised. 2026 is the golden age · expect more modalities (brain signals, weather, materials) by 2028.

</div>

---

# Consistency models · one-step diffusion

Song et al. 2023 · train a network $f_\theta(x_t, t) \to x_0$ such that the prediction is **consistent** along any trajectory from noise to data.

<div class="math-box">

Consistency property · $f_\theta(x_{t_1}, t_1) = f_\theta(x_{t_2}, t_2)$ for any $t_1, t_2$ on the same trajectory.

At inference · one forward pass from $x_T \to x_0$. No iteration.

</div>

<div class="realworld">

SDXL-Turbo (2023) · 1-step generation matching 50-step DDIM quality at real-time speeds. Latent Consistency Model (LCM) adapters · drop-in for Stable Diffusion. GAN-speed with diffusion-quality, finally.

</div>

---

# Flow matching · the next paradigm?

**Rectified flow** (Liu 2022) and **flow matching** (Lipman 2022) generalize diffusion:

- Train a network to predict a velocity field from noise to data.
- Sample with ODE solver in **4-10 steps** (vs diffusion's 50+).
- Cleaner math; often better quality at fewer steps.

Stable Diffusion 3, Flux, and many 2024+ models use flow-matching instead of pure diffusion. The boundary is blurring — both are "continuous-time generative models."

---

<!-- _class: summary-slide -->

# Lecture 22 — summary

- **Text conditioning** · CLIP text encoder + cross-attention in every U-Net block.
- **CFG** · $\epsilon_\text{CFG} = \epsilon(x, \emptyset) + w(\epsilon(x, c) - \epsilon(x, \emptyset))$. Default $w = 7$.
- **Latent diffusion** · VAE compresses 48×; U-Net diffuses in latent. Made SD feasible.
- **DDIM** · deterministic sampling, 20×+ fewer steps, no retraining.
- **DiT** · Transformer backbone replacing U-Net; powers Sora, SD3.
- **Flow matching** · successor to pure diffusion; fewer sampling steps, cleaner math.

### Read before Lecture 23

Chip Huyen blog posts on efficient inference; Dao 2022 (FlashAttention).

### Next lecture

**Efficient Inference** — KV-cache, quantization, distillation, FlashAttention, speculative decoding.

<div class="notebook">

**Notebook 22** · `22-stable-diffusion.ipynb` — use HF `diffusers`; implement CFG loop manually from a pretrained model; sweep guidance scale; compare samplers.

</div>
