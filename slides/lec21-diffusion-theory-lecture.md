---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Diffusion Models — Theory

## Lecture 21 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Where we are

- **VAE** (L19) · probabilistic encoder-decoder; good structure, blurry samples.
- **GAN** (L20) · sharp samples, unstable training, mode collapse.

Today · **diffusion**. Sharp samples + stable training + tractable likelihood. SOTA since 2021 for image, video, audio, 3D generation.

<div class="paper">

Today maps to **Prince Ch 18 (early)** + Ho et al. 2020 (DDPM) + Song &amp; Ermon 2020 (score-based).

</div>

Four questions:
1. What's the **forward process**?
2. What's the **closed-form** for $q(x_t \mid x_0)$?
3. How do we **train** a diffusion model?
4. What's the connection to **score matching**?

---

<!-- _class: section-divider -->

### PART 1

# Forward &amp; reverse · the big picture

Corrupt then learn to uncorrupt

---

# Forward corrupts · reverse reassembles

![w:920px](figures/lec21/svg/forward_reverse.svg)

<div class="realworld">

▶ Interactive: slide t, see a 2D spiral dissolve into noise; press "Reverse animate" to watch it reassemble — [diffusion-denoise](https://nipunbatra.github.io/interactive-articles/diffusion-denoise/).

</div>

---

<!-- _class: section-divider -->

### PART 2

# The forward process

Fixed · Markov · Gaussian

---

# One forward step

<div class="math-box">

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1 - \beta_t}\, x_{t-1}, \beta_t\, I\right)$$

- $\beta_t$ is a small positive scalar (the "noise schedule").
- Each step · shrink the signal slightly + add a little Gaussian noise.

</div>

Over $T$ steps (typically $T = 1000$), the data gradually washes out into pure $\mathcal{N}(0, I)$.

Forward is **not learned**. It's a fixed dynamical system designed to produce a tractable diffusion.

---

# The closed form · skip to any step

The magical property of Gaussian-noise addition: you can jump directly from $x_0$ to $x_t$ in one step.

<div class="math-box">

Define $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$. Then:

$$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1 - \bar{\alpha}_t)\, I\right)$$

Equivalently:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 \,+\, \sqrt{1 - \bar{\alpha}_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

</div>

No iteration needed during training. Sample $t$ uniformly, compute $x_t$ in closed form. Huge speedup.

---

# Noise schedules · in one chart

![w:920px](figures/lec21/svg/alpha_schedule.svg)

---

# Noise schedules · the numbers

| $t$ | Linear α̅_t | Cosine α̅_t | What's left |
|-----|-------------|-------------|-------------|
| 0 | 1.00 | 1.00 | clean signal |
| 250 | 0.62 | 0.82 | linear: 38% destroyed · cosine: 18% |
| 500 | 0.17 | 0.50 | linear: already mostly gone |
| 750 | 0.02 | 0.18 | linear: essentially pure noise |
| 1000 | 0.00 | 0.00 | both: N(0, I) |

<div class="insight">

Linear schedule wastes computation on steps near $T$ where everything is noise anyway. Cosine keeps the middle range useful — the middle is where the model actually learns.

</div>

---

# Noise schedules · linear vs cosine

Two common schedules:

**Linear** (original DDPM) · $\beta_t$ grows linearly from $10^{-4}$ to $0.02$ over $T = 1000$ steps.

**Cosine** (Nichol &amp; Dhariwal 2021) · $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$ — smoother, better for smaller $T$.

<div class="insight">

Cosine schedule adds noise more gradually at the start and faster at the end. Better quality at fewer diffusion steps. Used in most modern diffusion models.

</div>

---

<!-- _class: section-divider -->

### PART 3

# Training objective

Predict the noise

---

# The reverse process · parameterized

We want $p_\theta(x_{t-1} \mid x_t)$ — learn to denoise.

Ho et al. 2020 showed that if $q(x_{t-1} \mid x_t, x_0)$ is Gaussian (it is), the optimal reverse process is also Gaussian. So parameterize:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Further: parameterize to predict the **noise** $\epsilon$ rather than the mean directly. Simpler, better signal.

---

# DDPM loss · surprisingly simple

<div class="math-box">

$$\mathcal{L}_\text{DDPM} = \mathbb{E}_{t, x_0, \epsilon}\!\left[\,\left\| \epsilon \,-\, \epsilon_\theta\!\left(\sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon, \,t\right) \right\|^2\,\right]$$

</div>

In plain words:
1. Sample a clean image $x_0$ from the dataset.
2. Sample a timestep $t \in \{1, \ldots, T\}$.
3. Sample Gaussian noise $\epsilon$.
4. Compute $x_t$ (closed form).
5. Ask the network to predict $\epsilon$ from $(x_t, t)$.
6. MSE loss on the prediction.

**That's it.** Much simpler than GAN minimax or VAE ELBO.

---

<!-- _class: code-heavy -->

# DDPM in PyTorch · 30 lines

```python
def ddpm_loss(model, x0, T=1000):
    B = x0.size(0)
    t = torch.randint(0, T, (B,), device=x0.device)
    noise = torch.randn_like(x0)

    alpha_bar = alpha_bar_schedule[t].view(B, 1, 1, 1)
    x_t = alpha_bar.sqrt() * x0 + (1 - alpha_bar).sqrt() * noise

    pred_noise = model(x_t, t)                         # network input: noisy img + t
    return F.mse_loss(pred_noise, noise)

def sample(model, shape, T=1000):
    x = torch.randn(shape)                              # start from N(0, I)
    for t in reversed(range(T)):
        alpha_t     = alpha_schedule[t]
        alpha_bar_t = alpha_bar_schedule[t]
        predicted   = model(x, torch.tensor([t]))
        mean = (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * predicted) / alpha_t.sqrt()
        if t > 0:
            x = mean + alpha_t.sqrt() * torch.randn_like(x)
        else:
            x = mean
    return x
```

The network architecture is a **U-Net** (L9) with time-step conditioning injected into each block.

---

# Network architecture · in one picture

![w:920px](figures/lec21/svg/unet_time_conditioning.svg)

---

# Network architecture · U-Net with time

A diffusion model's $\epsilon_\theta(x_t, t)$ is typically a U-Net:

- Encoder downsamples, decoder upsamples.
- Skip connections between matching resolutions (from L9).
- **Time embedding** · $t$ becomes a sinusoidal vector, projected, and added into every block.
- **Attention** at low spatial resolutions (globally mix features).

<div class="realworld">

For 512×512 images · ~1B param U-Net; ~50 steps of sampling; ~5 seconds on a single GPU. Stable Diffusion's architecture is a direct descendant.

</div>

---

<!-- _class: section-divider -->

### PART 4

# Connection to score matching

Same thing, different derivation

---

# The score field in one picture

![w:920px](figures/lec21/svg/score_field.svg)

---

# The score function

Define the **score** of a distribution as $\nabla_x \log p(x)$ — the direction toward higher density.

If we had the score, we could do **Langevin dynamics** to sample:

$$x_{k+1} = x_k + \eta\, \nabla_x \log p(x_k) + \sqrt{2\eta}\, \xi, \quad \xi \sim \mathcal{N}(0, I)$$

This resembles the reverse diffusion process — follow the score, add a little noise.

---

# Diffusion ≈ score matching

Song &amp; Ermon 2020 (NCSN) showed: training $\epsilon_\theta(x_t, t)$ to predict noise **is equivalent** to training $s_\theta(x_t, t)$ to estimate the score $\nabla_x \log q(x_t)$, up to a constant.

<div class="math-box">

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

</div>

DDPM (Ho 2020) and score-SDE (Song 2020) are two lenses on the same model. Pick whichever you find more intuitive. In 2026 the DDPM formulation dominates for practical reasons (cleaner training recipe).

---

<!-- _class: section-divider -->

### PART 5

# Why diffusion won

---

# Diffusion vs VAE vs GAN

| | VAE | GAN | Diffusion |
|--|-----|-----|-----------|
| Sample quality | blurry | sharp | **SOTA** |
| Training | stable, fast | brittle | stable, slow |
| Likelihood | ELBO | ✗ | ELBO (loose) |
| Sampling | 1 forward | 1 forward | **T forwards** |
| Mode coverage | strong | mode collapse risk | **strong** |
| Interpretability | structured latent | messy latent | uniform latent |

<div class="keypoint">

Diffusion's big cost · slow sampling. This is what L22 will focus on — classifier-free guidance, latent diffusion, DDIM.

</div>

---

# Applications · 2026 state

- **Text-to-image** · Stable Diffusion, Midjourney, DALL-E 3, Imagen.
- **Video** · Sora, Runway Gen-3, VEO.
- **Audio** · AudioGen, Riffusion.
- **Molecule design** · RFdiffusion for proteins.
- **Robotics policies** · diffusion policy (Chi et al. 2023).

Diffusion has become the default generative model across modalities.

---

<!-- _class: summary-slide -->

# Lecture 21 — summary

- **Forward process** · add small Gaussian noise over T steps; closed form $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$.
- **Reverse process** · neural net predicts the noise; subtract step by step.
- **DDPM loss** · MSE between true noise and predicted noise. Stable, simple.
- **Schedule** · linear or cosine; cosine is modern default.
- **Architecture** · U-Net with sinusoidal time embedding + attention at low res.
- **Score matching** · same model through a different lens; reverse diffusion ≈ Langevin dynamics along the score.

### Read before Lecture 22

Prince Ch 18 (later sections) + HF `diffusers` docs + Rombach 2022 (Stable Diffusion).

### Next lecture

**Diffusion Models — Practice** — classifier-free guidance, latent diffusion, DDIM, DiT.

<div class="notebook">

**Notebook 21** · `21-ddpm-2d.ipynb` — implement DDPM on a 2D toy dataset (Swiss roll); visualize forward noising + reverse denoising animations.

</div>
