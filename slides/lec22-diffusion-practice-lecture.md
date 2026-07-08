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
6. Place **flow matching** and **consistency models** as the current frontier.

---

# Where we are · from toy DDPM to Stable Diffusion

Last lecture · **DDPM** — forward noise, learn reverse, predict ε. Works on MNIST, toy 2D.

<div class="insight">

**The problem with naive pixel-space diffusion.** L21's recipe has two dealbreakers. (1) It draws *whatever* it likes — you **can't steer it to a text prompt**. (2) It denoises all $512\times512\times3 \approx 800\text{k}$ pixels for ~1000 steps — **far too expensive** to run. Fix both and you get Stable Diffusion.

</div>

Today's three fixes — each one a concrete **object you can name**:

1. **Cross-attention conditioning** — the U-Net *reads a text vector*, so a prompt can steer it.
2. **Classifier-free guidance** — run the model *twice* (prompt / no-prompt) and push along the difference, so the prompt actually **sticks**.
3. **Latent diffusion** — run the whole loop in a *tiny VAE latent*, not pixels, so it's **cheap**.

Then **DDIM** cuts the 1000 steps → ~50. Snap these together → **Stable Diffusion**.

<div class="paper">

**Reading & inspiration** · **Rombach et al. 2022 *Stable Diffusion (LDM)*** · **Ho & Salimans 2022 *Classifier-Free Guidance*** · **Song et al. 2021 *DDIM*** (fast sampling) · **Hugging Face `diffusers` docs** (code-first) · **Lilian Weng** diffusion blog (later sections) · **UDL (Prince) Ch 18 (later)**.

</div>

---

# Pop quiz · how does "an astronaut riding a horse" become an image?

L21 gave us *unconditional* diffusion · noise → image.

<div class="popquiz">

(a) The text is fed into the U-Net as a separate channel.
(b) A frozen text encoder (CLIP) maps the prompt to a vector · the U-Net **cross-attends** to that vector at every layer.
(c) The text is tokenized and prepended as image patches.

Stop and decide. **Answer · (b)** — and the magic that lets the same network draw "a corgi in a tuxedo" or "a slice of pizza" is just *changing the prompt vector at inference*. Today's lecture is the engineering that makes this work.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Conditioning · from unconditional to text-to-image

How does the prompt actually steer the denoiser?

---

# How to condition a diffusion model

We want $\epsilon_\theta(x_t, t, c)$ — predict noise given a **condition** $c$ (class label, text embedding, image).

<div class="keypoint">

**Big idea in one line.** Give the denoiser a *third input* $c$. Everything below is just *how* $c$ gets in — Stable Diffusion picks the last option, cross-attention.

</div>

Three common ways to inject $c$:

1. **Concatenate** $c$ to the input channels (for class labels).
2. **Add** $c$ to the time embedding (simple but loses detail).
3. **Cross-attention** between $c$ and intermediate features (Stable Diffusion, most modern).

---

# Cross-attention · how an image listens to text

<div class="insight">

**Analogy · team of painters.** Each painter handles a small patch. Before painting they ask: *"which word in the prompt is most relevant to **my patch**?"* The painter near the top pays attention to "hat"; the furry-patch painter pays attention to "cat".

</div>

The mechanism — for each image region:
1. **Get word vectors** from a frozen text encoder (CLIP) → keys $K$ and values $V$.
2. **Get region "questions"** from the U-Net's intermediate features → queries $Q$.
3. **Score relevance** · $\text{sim}(Q, K)$, then softmax.
4. **Final instruction** · weighted sum of $V$ → used to denoise that region.

---

<!-- _class: code-heavy -->

# Cross-attention · in code

```python
text_emb = clip_text_encoder("a cat astronaut")  # [1, 77, 768]

class CrossAttention(nn.Module):
    def forward(self, spatial_features, text_emb):
        Q = self.q_proj(spatial_features)
        K = self.k_proj(text_emb)
        V = self.v_proj(text_emb)
        return softmax(Q @ K.T / sqrt(d)) @ V
```

---

# Cross-attention · worked numeric

Prompt: "a red square". Two word vectors: $V_\text{red} = [1, 0]$, $V_\text{square} = [0, 1]$.

Top-left pixel's current state: $Q_\text{TL} = [0.1, 0.9]$ (more square-like than red-like).

**Step 1 · scores.**
- $\text{sim}(Q, V_\text{red}) = 0.1 \cdot 1 + 0.9 \cdot 0 = 0.1$
- $\text{sim}(Q, V_\text{square}) = 0.1 \cdot 0 + 0.9 \cdot 1 = 0.9$

"Square" is much more relevant.

**Step 2 · softmax → weights.** Suppose $w_\text{red} = 0.3$, $w_\text{square} = 0.7$.

**Step 3 · final instruction.**
$0.3 \cdot [1, 0] + 0.7 \cdot [0, 1] = [\mathbf{0.3, 0.7}]$

Used to update this pixel — **slightly more red, much more square**.

---

# Cross-attention · why it works for text conditioning

Self-attention inside the U-Net mixes spatial positions. **Cross**-attention *between spatial features and text features* is where the prompt enters.

<div class="keypoint">

At a high-resolution block · each pixel asks "which prompt tokens matter for me?". A cat pixel pays attention to "cat" in the prompt; a sky pixel to "sky". The result is **spatially localized conditioning** — a single prompt can steer different regions differently.

</div>

Visualizing the cross-attention maps reveals exactly this — each word has a blob of pixels that listened most to it. This is the mechanism behind DreamBooth, Prompt-to-Prompt, and every prompt-editing technique.

<div class="realworld">

▶ Interactive: type a prompt, watch text tokens steer the denoising — [text-diffusion](https://nipunbatra.github.io/interactive-articles/text-diffusion/).

</div>

---

<!-- _class: section-divider -->

### PART 2

# Classifier-Free Guidance

Conditioning is in — so why do samples still half-ignore the prompt?

---

# CFG · the big idea, then the geometry

<div class="keypoint">

**Object first.** Each step the model runs **twice** and outputs two noise predictions — $\epsilon_\text{cond}$ (*with* the prompt) and $\epsilon_\text{uncond}$ (*no* prompt). We **extrapolate** along their difference; the guidance scale $w$ is simply *how far* we walk:
$$\epsilon_\text{cfg} = \epsilon_\text{uncond} + w\,(\epsilon_\text{cond} - \epsilon_\text{uncond})$$

</div>

![w:800px](figures/lec22/svg/cfg_guidance_vectors.svg)

---

# CFG · the two-compass analogy

<div class="insight">

You're lost in a forest with two compasses.
1. **Generic compass** · always points to "average image" (unconditional prediction $\epsilon_\emptyset$).
2. **Special compass** · points toward "cabin in the woods" (prompt-conditional $\epsilon_c$).

The **difference vector** $\epsilon_c - \epsilon_\emptyset$ is the **pure influence of the prompt**. CFG · start at the generic point and walk $w$ times that vector → over-shoot the conditional prediction in the prompt direction.

</div>

---

# ⭐⭐⭐ Optional · CFG · derive the formula step by step

<div class="keypoint">

**Gist (skip the rest if you're happy).** The formula just says *"start at the no-prompt prediction, walk $w$ steps toward the prompt."* Below is what each piece is.

</div>

1. **Generic direction.** $\epsilon_\text{uncond} = \epsilon_\theta(x_t, \emptyset)$ (null prompt).
2. **Prompt direction.** $\epsilon_\text{cond} = \epsilon_\theta(x_t, c)$.
3. **Pure prompt influence.** $\Delta = \epsilon_\text{cond} - \epsilon_\text{uncond}$.
4. **Extrapolate** by guidance scale $w$:
$$\epsilon_\text{CFG} = \epsilon_\text{uncond} + w \cdot \Delta = \epsilon_\theta(x_t, \emptyset) + w\bigl(\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset)\bigr)$$

**Read the dial** · $w = 0$ ignore prompt · $w = 1$ exact conditional · $w > 1$ overshoot to **amplify** adherence. **Default $w = 7.5$** in Stable Diffusion.

---

# CFG in pictures · same prompt, only $w$ changes

Same prompt, same seed — we only turn the $w$ dial. Adherence rises, then colors over-saturate.

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
| 7.5 | Stable Diffusion default · balanced |
| 12+ | strong adherence · saturated colors, artifacts |
| 25+ | cartoonish oversaturation; often broken |

<div class="insight">

CFG trades **diversity for prompt adherence**. Low $w$ produces many different interpretations; high $w$ produces a narrower, more on-prompt distribution. Treat $w$ as a hyperparameter you sweep for each task.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Latent diffusion

What made Stable Diffusion cheap enough to ship?

---

# Why diffuse in latent space · the intuition

<div class="keypoint">

**Object first.** Run the *entire* diffusion loop inside a VAE's **$64\times64\times4$ latent**, not on pixels. The U-Net denoises the latent; a VAE **decoder** turns the final latent into the $512\times512$ image — once, at the very end.

</div>

Most pixels in an image are *correlated*. A patch of blue sky has hundreds of near-identical pixel values. Diffusing each one independently is wasteful.

<div class="keypoint">

A pretrained VAE has already absorbed the **perceptually redundant** structure · the latent is a near-minimal representation. Diffusion then only has to model the *interesting* (high-entropy) dimensions.

</div>

Result · 48× fewer dimensions, same perceptual quality, ~10× faster sampling. This is the single change that made Stable Diffusion runnable on consumer GPUs.

---

# Latent diffusion · the zip-the-file analogy

<div class="insight">

A 512×512 photo of blue sky has 262,144 pixels — most are nearly identical shades of blue. Asking a network to predict "this blue pixel is followed by a blue pixel" 1000× is wasteful.

**Like emailing a 100MB doc.** Instead of attaching it raw, you zip to 1MB. The zip captures the important info in less space. Latent diffusion does the same · zip the image with a VAE, run diffusion in the small zipped space, then unzip at the end.

</div>

---

# The dimension math

512×512 RGB = $512 \cdot 512 \cdot 3 = \mathbf{786{,}432}$ dimensions.

Stable Diffusion's VAE compresses to 64×64×4 = $64 \cdot 64 \cdot 4 = \mathbf{16{,}384}$ dimensions.

**Saving · $786{,}432 / 16{,}384 = \mathbf{48\times}$ fewer dimensions** for the expensive U-Net to process.

The diffusion loop runs in latent space, **starting from latent noise**; the VAE **decoder** runs once at the end to render the image. (The encoder is used in *training* and for img2img — not for text-to-image sampling, where there's no input image to encode.) Single biggest reason Stable Diffusion is shippable on consumer GPUs.

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

<div class="insight">

**Now it clicks — that's Stable Diffusion.** CLIP text encoder → U-Net that cross-attends to the prompt (with CFG each step), diffusing in the VAE latent → VAE decoder renders the image. Three fixes, one pipeline.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Faster sampling

1000 forward passes per image — can we skip most of them?

---

# Why 1000 steps · the quick math

<div class="keypoint">

**Object first · what's a sampler?** Just the *rule for how big a denoising step to take*. **DDPM** takes 1000 tiny random steps; **DDIM** takes ~50 big **deterministic** steps from the *same* trained model. First, why does DDPM need so many?

</div>

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

# ⭐⭐⭐ Optional · DDIM · deterministic sampling in 20–50 steps

<div class="keypoint">

**Gist.** DDPM's 1000 steps assume each step is *random* (Markovian). DDIM notices you can make the steps **deterministic** and then *jump over* most of them — ~50 steps, same quality, same weights.

</div>

<div class="paper">

Song, Meng, Ermon 2020 · *"Denoising Diffusion Implicit Models"*

</div>

DDIM reformulates the reverse process to be **deterministic** given the initial noise. Crucially, the trained DDPM model can be sampled with DDIM — no retraining.

**Effect** · 50 DDIM steps ≈ 1000 DDPM steps in quality. **20× speedup** essentially for free.

As of this writing, DDIM (and its successor DPM-Solver++) is the default sampler in most diffusion libraries.

---

# DiT · the global town-hall analogy

<div class="insight">

**Convolution (U-Net):** you only talk to your neighbours. Messages travel via "telephone" through many layers to reach a far-away pixel.

**Attention (Transformer):** every pixel is in a global video call. The cat-ear pixel can directly ask the cat-tail pixel "are you also part of the cat?" Better long-range understanding.

</div>

---

# DiT · how it works

Peebles & Xie 2023 · replace the U-Net with a Transformer.

1. **Patchify** the (small, latent) noisy image into a grid of patches (like ViT).
2. **Treat patches as tokens** · flatten each → vector. Image becomes a *sequence*.
3. **Transformer blocks** · self-attention lets every patch look at every other.
4. **Condition** on time + class/text via **adaLN** (adaptive LayerNorm).
5. **Re-assemble** → noise prediction.

**Why** · Transformers scale incredibly well — more data + compute = better samples where U-Net plateaus — and DiT inherits the LLM ecosystem (FlashAttention, tensor parallelism). Backbone of **Sora**, **Stable Diffusion 3**, and most 2024+ frontier image/video models.

---

<!-- _class: section-divider -->

### PART 5

# The generative landscape

Where does all of this stand today — and what comes after diffusion?

---

# What ships · as of this writing

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

# Diffusion · by modality

<div class="math-box">

| Modality | Model | Speed to generate |
|:-:|:-:|:-:|
| Images 1024² | SD3 · Flux | ~0.5 images/s (20 steps) |
| Video 720p · 16s | Sora · VEO | minutes per clip |
| Audio music | Stable Audio · AudioLDM | seconds per clip |
| 3D meshes | Diffusion-SDF · ShapE | ~30s per mesh |
| Molecules | RFdiffusion (Baker lab — Baker shared the 2024 Chemistry Nobel for protein design) | ~10s per protein |
| Robot policies | Diffusion Policy (Chi 2023) | 50 Hz control loop |

</div>

<div class="insight">

The diffusion paradigm scaled to every signal that can be noised. The one big exception · **text**, where autoregressive LLMs still win — diffusion is one of the two big generative paradigms, not the only one.

</div>

---

# Consistency models · winding road vs teleporter

<div class="insight">

Normal DDIM/DDPM · walk down a winding mountain path from foggy peak (noise) to clear valley (image). 50+ steps along the path.

**Consistency model** = a teleporter. Stand at any point on the path, press a button, instantly arrive at the valley. The teleporter is *consistent* — no matter where on the path you stand, it always takes you to the same final $x_0$.

</div>

---

# Consistency models · the property

Train $f_\theta(x_t, t) \to x_0$ to satisfy:
$$f_\theta(x_{t_1}, t_1) = f_\theta(x_{t_2}, t_2) \approx x_0$$
for any two points on the same trajectory.

**Inference:**
1. Sample $x_T \sim \mathcal{N}(0, I)$.
2. Run network **once**: $\hat x_0 = f_\theta(x_T, T)$.
3. Done.

SDXL-Turbo (2023) · 1-step generation at near-50-step DDIM quality. LCM adapters · drop-in for Stable Diffusion. **GAN-speed with diffusion-quality, finally.**

---

# Flow matching · the next paradigm?

**Rectified flow** (Liu 2022) and **flow matching** (Lipman 2022) generalize diffusion:

- Train a network to predict a velocity field from noise to data.
- Sample with ODE solver in **4-10 steps** (vs diffusion's 50+).
- Cleaner math; often better quality at fewer steps.

Stable Diffusion 3, Flux, and many 2024+ models use flow-matching instead of pure diffusion. The boundary is blurring — both are "continuous-time generative models."

---

# The sampler menu · as of this writing

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

5-year trajectory · 1000 steps (2020) → 50 steps (DDIM) → 4 steps (consistency) → 1 step (distilled rectified flow). Each generation cut inference cost by ~10×.

</div>

---

<!-- _class: summary-slide -->

# Putting it all together · the L22 master sentence

<div class="math-box">

**Three engineering tricks turn DDPM into Stable Diffusion** · (1) **conditioning** via cross-attention to a CLIP text vector · (2) **classifier-free guidance** to amplify the prompt signal · (3) **latent diffusion** — run the whole process in a $4\times64\times64$ VAE latent instead of $3\times512\times512$ pixels. **DDIM** then drops the steps from 1000 → 25–50 by reusing the same trained model with a deterministic sampler. **DiT** swaps the U-Net for a Transformer, scaling more cleanly.

</div>

Stable Diffusion 2022, SD3 / FLUX 2024, Sora 2024 — all run this same recipe with bigger backbones.

---

# The four tricks · cost vs benefit

| Trick | What it costs | What it buys |
|:-:|:-:|:-:|
| Cross-attn conditioning | a frozen text encoder | text → image |
| CFG (scale ~7.5) | doubled forward pass | sharper, more on-prompt |
| Latent diffusion (VAE) | a pretrained VAE | $48\times$ fewer dimensions to diffuse |
| DDIM | smarter sampling, no retrain | $20\times$ fewer steps |

---

# Practice problems · 1

<div class="math-box">

**P1.** Cross-attention from U-Net features (queries) to a 77-token CLIP text vector (keys, values). Sketch the shapes.

**P2.** **CFG** combines $\hat\epsilon_\text{cond}$ and $\hat\epsilon_\text{uncond}$ as $\hat\epsilon = \hat\epsilon_\text{uncond} + s(\hat\epsilon_\text{cond} - \hat\epsilon_\text{uncond})$. What does $s = 1$ produce? $s = 0$? $s = 7.5$?

**P3.** Stable Diffusion's VAE compresses $3 \times 512 \times 512 = 786{,}432$ pixels into $4 \times 64 \times 64 = 16{,}384$ latents. Compute the compression ratio. Why doesn't this destroy image quality?

</div>

---

# Practice problems · 2

<div class="math-box">

**P4.** **DDIM** lets you sample in 20–50 steps instead of 1000 using the *same* trained model. In one sentence, why is no retraining needed?

**P5.** Why does **DiT** (Transformer backbone) scale better than U-Net? Tie back to the L13 master sentence about Transformer scaling.

**P6.** Pick **Stable Diffusion / DiT / FLUX / Sora** for each of (a) $512\times512$ photographs, (b) $1024\times1024$ vector-style art, (c) 5-second video clips. Why?

</div>

---

# Lecture 22 — summary

- **Text conditioning** · CLIP text encoder + cross-attention in every U-Net block.
- **CFG** · $\epsilon_\text{CFG} = \epsilon(x, \emptyset) + w(\epsilon(x, c) - \epsilon(x, \emptyset))$. Default $w = 7.5$.
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

---

# The one-sentence takeaway

<div class="insight">

**Diffuse where the information lives — in latent space, not pixel space.**

*Next lecture: the model is trained; now make it cheap enough to serve.*

</div>
