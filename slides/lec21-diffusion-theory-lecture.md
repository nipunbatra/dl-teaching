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

# Learning outcomes

By the end of this lecture you will be able to:

1. Explain the **forward process** in one sentence and write it down.
2. Derive the closed-form $q(x_t \mid x_0)$ and use it in code.
3. Write the **DDPM training loss** and describe what each term is doing.
4. Describe the **reverse process** step-by-step.
5. Understand **noise schedules** (linear vs cosine) and pick one for a task.
6. Connect **DDPM** to **score matching** via Langevin dynamics.
7. State why diffusion won over GANs and VAEs for image/video/audio.

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

# The intuition in one sentence

<div class="keypoint">

**Gradually turn an image into pure noise, then train a network to reverse that process one tiny step at a time.**

</div>

At the end of training, you can start from random noise and reverse-diffuse it into a brand new image. Each small step is easy to learn; chained together they generate.

---

# A physical analogy · ink in water

Drop a drop of ink into a glass of water. It stays concentrated, then slowly spreads, then uniformly tints the water.

<div class="columns">
<div>

### Forward (easy)

Ink diffuses into water · we can describe this with a simple diffusion equation. Watching a drop blur is what "noise corrupts the signal" looks like in pictures.

</div>
<div>

### Reverse (hard)

"Un-diffuse" the ink back into a drop. Physics says impossible (entropy only grows). But with data · we have many *examples* of initial states. A neural network can learn the reverse direction from those examples.

</div>
</div>

Diffusion models learn the miracle "reverse" that physics doesn't give you — but they learn it from data, not first principles.

---

# Why this is better than GANs

<div class="columns">
<div>

### GAN problems

- minimax: two networks playing a game
- mode collapse
- unstable training
- hyper-sensitive to hyperparameters

</div>
<div>

### Diffusion advantages

- regression loss: MSE on predicted noise
- **one** network
- stable training
- default settings usually work

</div>
</div>

<div class="insight">

A GAN asks a network to hit a moving target (the discriminator's decision boundary). A diffusion model asks a network to match a *static* target (the noise that was added). Static targets are fundamentally easier to optimize.

</div>

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

# Why shrink *and* add noise

You might ask · why not just add noise? Why shrink the signal too?

<div class="keypoint">

Shrinking keeps the **total variance bounded**. If you only add noise, the variance grows without limit; $x_T$ would be impossibly noisy and nothing-like-$\mathcal{N}(0, I)$.

</div>

<div class="math-box">

Variance check · if $x_{t-1} \sim \mathcal{N}(0, I)$, then $x_t = \sqrt{1 - \beta_t}\, x_{t-1} + \sqrt{\beta_t}\, \epsilon$.

Variance of $x_t$ = $(1 - \beta_t) + \beta_t$ = **1**. Always.

</div>

This is why the forward process preserves unit variance — it's a **variance-preserving SDE**.

---

# A step-by-step · small β = 0.01

Start with $x_0 = 2.0$. Apply 5 forward steps with $\beta_t = 0.01$:

<div class="math-box">

| $t$ | $x_t$ | noise added |
|:-:|:-:|:-:|
| 0 | 2.00 | — |
| 1 | 1.99 · √0.99 + 0.1·ε = 1.97 + 0.08 = 2.05 | ε = 0.8 |
| 2 | 2.02 | ε = -0.4 |
| 3 | 2.00 | ε = -0.2 |
| 4 | 2.03 | ε = 0.3 |
| 5 | 2.06 | ε = 0.2 |

</div>

After 5 steps the signal is barely disturbed. After 1000 steps with growing β, it becomes standard normal. The *accumulated* effect, not each step, turns signal into noise.

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

# Why closed-form matters · training speed

<div class="columns">
<div>

### Naive (iterative) forward

To get $x_{500}$, you'd apply 500 Gaussian steps sequentially. 500× forward passes per training example.

Batch of 128, 100k examples · ~10⁹ operations just to make noise targets. Days on a single GPU.

</div>
<div>

### Closed-form

One sample of $\epsilon$, one scaled add. 500× faster per example.

Batch of 128 in **one step** · microseconds. Hours instead of days.

</div>
</div>

<div class="keypoint">

This closed-form is the single biggest practical advantage over continuous-time score-SDE approaches. Without it, DDPM training would cost 500× more.

</div>

---

# Closed-form · the derivation in 3 lines

<div class="math-box">

Start from one step: $x_t = \sqrt{\alpha_t}\, x_{t-1} + \sqrt{1 - \alpha_t}\, \epsilon_t$.

Unroll: $x_t = \sqrt{\alpha_t}[\sqrt{\alpha_{t-1}} x_{t-2} + \sqrt{1-\alpha_{t-1}}\epsilon_{t-1}] + \sqrt{1-\alpha_t}\epsilon_t$

Merge Gaussians (sum of independent Gaussians = Gaussian with summed variances):

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \bar\epsilon$$

</div>

where $\bar\epsilon \sim \mathcal{N}(0, I)$ replaces the $t$-step chain of independent $\epsilon$'s. Gaussian closed under convolution — this is the magic.

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

# Picking T · the hyperparameter most people ignore

| $T$ | Behavior |
|:-:|:-:|
| 50 | too coarse; each step must learn a big jump; sample quality hurts |
| 200 | works but poor quality at the extremes |
| 1000 | **default**; great quality with cosine schedule |
| 4000 | slight quality gain; 4× inference cost; rarely worth it |

<div class="realworld">

DDPM (Ho 2020) used T=1000 with linear schedule. Nichol &amp; Dhariwal 2021 showed cosine + T=4000 gave marginal gains; T=1000+cosine is today's sweet spot.

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

# Why predict $\epsilon$ instead of the mean?

Given $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$, there are three equivalent prediction targets:

<div class="math-box">

- Predict $x_0$ · known as "x0-prediction" or "v-prediction variant"
- Predict $\mu_\theta(x_t, t)$ · the mean of the reverse distribution
- Predict $\epsilon$ · the noise that was added

</div>

Ho et al. 2020 showed **$\epsilon$-prediction gives the best sample quality**. Intuition · noise is unit-variance and dimension-independent; the network doesn't need to learn the scale of the signal.

Modern models (SDXL, Imagen) often use "v-prediction" — a weighted combination that's more numerically stable at small $t$.

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

# Worked example · one training step

<div class="math-box">

Suppose $x_0 = [2.0, 1.0]$ (a 2D data point), $t = 500$, $\bar\alpha_{500} = 0.5$. Sample $\epsilon = [0.3, -0.2]$.

1. $x_{500} = \sqrt{0.5} \cdot [2.0, 1.0] + \sqrt{0.5} \cdot [0.3, -0.2] = [1.414, 0.707] + [0.212, -0.141]$
   = $[1.626, 0.566]$
2. Feed $(x_{500}, t=500)$ to the network. Prediction · $\hat\epsilon = [0.25, -0.15]$
3. Loss · $\|\hat\epsilon - \epsilon\|^2 = (0.25-0.3)^2 + (-0.15 + 0.2)^2 = 0.005$
4. Backprop through $\hat\epsilon$ to update the network.

</div>

Single example, single $t$. Sum the loss over a batch and all is ready. No adversarial game, no multiple networks, no cross-entropy.

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

# Reverse step · what's happening

Given $x_t$, the reverse step computes:

<div class="math-box">

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon_\theta(x_t, t)\right) + \sigma_t\, z$$

where $z \sim \mathcal{N}(0, I)$ (added noise) and $\sigma_t$ is the reverse variance.

</div>

Decoded:
- **First term** · estimate of the clean signal (un-scale the noise prediction, subtract from $x_t$).
- **Second term** · re-inject fresh noise at scale $\sigma_t$ so the chain stays stochastic.

At $t = 1$, we drop the noise term — deterministic final step.

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

# Sinusoidal time embedding

The time $t$ is an integer $\in \{1, \ldots, T\}$. Represent it as a dense vector using the same positional encoding from L13:

```python
def timestep_embedding(t, d):
    half = d // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half).float() / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([args.cos(), args.sin()], dim=-1)
```

<div class="insight">

The same reason as in Transformers (L13) · sinusoidal basis gives multi-scale time representation that the network can read at any scale. Learned embeddings work too; sinusoidal is more robust across training-time changes in $T$.

</div>

---

# Time conditioning · inject at every block

```python
class TimestepBlock(nn.Module):
    def forward(self, x, t_emb):
        # x: image features. t_emb: timestep embedding
        h = self.norm1(x)
        h = self.conv1(F.silu(h))
        # project time and add as bias (broadcast over spatial dims)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h
```

Each U-Net residual block receives the time embedding and adds it to the channel dimension. The **same network weights** handle all timesteps — time is just another input, not a different model per step.

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

# Score vs density · why use the score?

<div class="columns">
<div>

### Density $p(x)$

- Must be non-negative.
- Must integrate to 1.
- Intractable normalizing constant for complex distributions.

Hard to model with a neural network.

</div>
<div>

### Score $\nabla_x \log p(x)$

- Any vector field.
- Normalizer disappears: $\nabla_x \log (p \cdot Z) = \nabla_x \log p$.
- Easy to model with a neural network.

Parametrize the *derivative*, not the function itself. Samples are what we want anyway.

</div>
</div>

<div class="keypoint">

Modeling the score sidesteps the normalizer problem — and the score is exactly what you need to run Langevin sampling.

</div>

---

# Diffusion ≈ score matching

Song &amp; Ermon 2020 (NCSN) showed: training $\epsilon_\theta(x_t, t)$ to predict noise **is equivalent** to training $s_\theta(x_t, t)$ to estimate the score $\nabla_x \log q(x_t)$, up to a constant.

<div class="math-box">

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

</div>

DDPM (Ho 2020) and score-SDE (Song 2020) are two lenses on the same model. Pick whichever you find more intuitive. In 2026 the DDPM formulation dominates for practical reasons (cleaner training recipe).

---

# Two views side-by-side

<div class="columns">
<div>

### DDPM view (Ho 2020)

- Forward · fixed Markov chain.
- Reverse · learned Gaussian chain.
- Loss · MSE between predicted and true noise.
- Intuition · *denoising* at multiple scales.

</div>
<div>

### Score-SDE view (Song 2020)

- Forward · SDE driving data to noise.
- Reverse · another SDE driving noise to data.
- Loss · score matching.
- Intuition · gradient field pointing to data.

</div>
</div>

<div class="insight">

Use whichever is easier for your problem. DDPM's discrete-time recipe is simpler to code; Score-SDE gives more flexibility for continuous-time / arbitrary-schedule models (e.g., flow matching in 2023+).

</div>

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

# Why diffusion beat GANs on image quality

1. **Training signal is always strong** · MSE on noise has a meaningful gradient at every step and every $x_t$. GAN's adversarial loss often gives near-zero gradient early.
2. **No mode collapse** · every training example teaches the model to denoise independently. The model can't "cheat" by producing one output.
3. **Iterative refinement** · generation is 50-1000 tiny corrections. Errors at each step are small; the chain self-corrects. GANs must produce the final output in one forward pass.
4. **Infinite data augmentation** · every (x₀, t, ε) triple is a new training example. A dataset of 10k images gives you a virtually infinite training stream.

---

# A picture of why iteration helps

Think about drawing a face. A GAN must commit · "these pixels are skin, these are eyes, this is hair" — all in one forward pass. Wrong commitments cascade.

<div class="keypoint">

Diffusion starts with pure noise; the *first* reverse step sketches the rough layout; the *second* adds features; the *hundredth* adds skin texture. The network revises its answer 1000 times, getting it right in the limit.

</div>

This is why diffusion samples look sharper and more coherent than any single-forward-pass generator.

---

# Applications · 2026 state

- **Text-to-image** · Stable Diffusion, Midjourney, DALL-E 3, Imagen.
- **Video** · Sora, Runway Gen-3, VEO.
- **Audio** · AudioGen, Riffusion.
- **Molecule design** · RFdiffusion for proteins.
- **Robotics policies** · diffusion policy (Chi et al. 2023).

Diffusion has become the default generative model across modalities.

---

# Frontier · where diffusion is heading

<div class="columns">
<div>

### Faster sampling

- DDIM (L22) · deterministic, 20 steps.
- Flow matching · 5-10 steps.
- Consistency models · 1-4 steps.

</div>
<div>

### Richer conditioning

- CFG (L22) · text steering.
- ControlNet · per-pixel conditioning (pose, depth).
- Inpainting · mask what to regenerate.

</div>
</div>

<div class="realworld">

Consistency models and flow matching are closing the "slow sampling" gap. In 2026 · expect 1-step diffusion samplers to become competitive with GANs on speed.

</div>

---

# Common questions · FAQ

**Q. Is diffusion a likelihood-based model?**
A. Yes, approximately. The DDPM loss corresponds to a variational lower bound on $\log p(x)$, but with a specific weighting. Tight bounds need "improved DDPM" tricks.

**Q. Why is the schedule Gaussian, not uniform?**
A. Because Gaussians are closed under convolution — lets us write $q(x_t \mid x_0)$ in closed form. Other noise distributions (uniform, Laplacian) don't give this gift.

**Q. What if the data isn't image-like?**
A. Use a different architecture (Transformer for sequences, GNN for graphs). The diffusion recipe is independent of architecture — only the noise-prediction network changes.

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
