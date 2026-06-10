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

**Reading & inspiration** · **Lilian Weng, *What are Diffusion Models?*** (the canonical walk-through) · **Calvin Luo, *Understanding Diffusion Models*** (arXiv 2022) · **Ho et al. 2020 *DDPM*** + **Song & Ermon 2020** (score-based) · **Sohl-Dickstein et al. 2015** (the original idea) · **UDL (Prince) Ch 18 (early)**.

</div>

Four questions:
1. What's the **forward process**?
2. What's the **closed-form** for $q(x_t \mid x_0)$?
3. How do we **train** a diffusion model?
4. What's the connection to **score matching**?

---

# Pop quiz · how would *you* generate images?

You have 1 million natural images. You want a model that draws new ones.

<div class="popquiz">

(a) Train a *very large* CNN to map noise → image directly (one pass).
(b) Add a *little* noise at a time over $T$ steps, then learn to **reverse one step at a time**.
(c) Memorize and interpolate.

The miraculous answer · **(b)**. Predicting *one step* of denoising is **vastly easier** than going from pure noise to an image in one pass — the same way drawing a line is easier than drawing the whole sketch in one stroke.

This single re-framing — *"break a hard generative problem into many easy denoising problems"* — is why diffusion won.

</div>

---

<!-- _class: section-divider -->

### PART 1

# Forward &amp; reverse · the big picture

How can *destroying* data teach a network to *create* it?

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

---

# Forward + reverse · sequence view

![w:920px](figures/lec21/svg/diffusion_timeline.svg)

<div class="realworld">

▶ Interactive: slide t, see a 2D spiral dissolve into noise; press "Reverse animate" to watch it reassemble — [diffusion-denoise](https://nipunbatra.github.io/interactive-articles/diffusion-denoise/).

</div>

---

<!-- _class: section-divider -->

### PART 2

# The forward process

What exactly does one noising step do? Fixed · Markov · Gaussian

---

# One forward step · slightly blurring a photo

<div class="insight">

**Analogy.** Take a sharp photo and make it one step blurrier:
1. Fade the original a tiny bit (e.g. to 99.5% opacity).
2. Add a faint layer of random static (Gaussian noise).

That's it. $\beta_t$ controls both the fade amount and the static intensity.

</div>

Formally:
$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\ \sqrt{1 - \beta_t}\,x_{t-1},\ \beta_t\,I\right)$$

i.e. $x_t = \sqrt{1-\beta_t}\,x_{t-1} + \sqrt{\beta_t}\,\epsilon$ with $\epsilon \sim \mathcal{N}(0, 1)$.

---

# One forward step · worked numeric

$x_{t-1} = 100$, $\beta_t = 0.01$.
- Fade · $\sqrt{0.99} \approx 0.995$, so mean = $99.5$.
- Noise · std $= \sqrt{0.01} = 0.1$.
- Update · $x_t = 0.995 \cdot 100 + 0.1 \cdot \epsilon$.

Over $T = 1000$ steps, the signal washes out into pure $\mathcal{N}(0, I)$. Forward is **not learned** — fixed dynamical system.

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
| 1 | 2.00 · √0.99 + 0.1·ε = 1.99 + 0.08 = 2.07 | ε = 0.8 |
| 2 | 2.02 | ε = -0.4 |
| 3 | 1.99 | ε = -0.2 |
| 4 | 2.01 | ε = 0.3 |
| 5 | 2.02 | ε = 0.2 |

</div>

After 5 steps the signal is barely disturbed. After 1000 steps with growing β, it becomes standard normal. The *accumulated* effect, not each step, turns signal into noise.

---

# The closed form · compounding fades

After 1 step, signal is faded by $\sqrt{\alpha_1}$. After 2 steps · $\sqrt{\alpha_2 \alpha_1}$. After $t$ steps · $\sqrt{\bar\alpha_t} = \sqrt{\prod_{s=1}^t \alpha_s}$.

All the per-step noise additions also "pool together" into one big Gaussian:
$$\boxed{x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, I)}$$

Variance check · we designed the process so total variance stays 1. If $\bar\alpha_t$ is the fraction left from the signal, $1 - \bar\alpha_t$ is the fraction from noise. **Sums to 1.**

---

# Closed form · worked numeric · jump straight to $x_{500}$

$x_0 = 2.0$, and the linear DDPM schedule gives $\bar\alpha_{500} \approx 0.08$.
- Signal scale · $\sqrt{0.08} \approx 0.283$.
- Noise scale · $\sqrt{0.92} \approx 0.959$.
- Sample $\epsilon = -0.8$.
- $x_{500} = 0.283 \cdot 2.0 + 0.959 \cdot (-0.8) = 0.566 - 0.767 = \mathbf{-0.20}$.

After 500 steps the original signal of 2.0 has nearly washed away — $x_{500}$ is mostly noise. **No iteration needed during training.**

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

# Closed-form · the derivation in 3 lines · the one to know

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
| 250 | 0.52 | 0.85 | linear: 48% destroyed · cosine: 15% |
| 500 | 0.08 | 0.49 | linear: already mostly gone |
| 750 | 0.003 | 0.14 | linear: essentially pure noise |
| 1000 | 0.00 | 0.00 | both: ≈ N(0, I) |

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

What should the network predict — and with what loss?

---

# The reverse process · parameterized

We want $p_\theta(x_{t-1} \mid x_t)$ — learn to denoise.

The posterior $q(x_{t-1} \mid x_t, x_0)$ is exactly Gaussian, and for small $\beta_t$ the true reverse step is well-approximated by a Gaussian too. So parameterize:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Further: parameterize to predict the **noise** $\epsilon$ rather than the mean directly. Simpler, better signal.

---

# Why predict $\epsilon$ instead of the mean?

Given $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon$, there are three equivalent prediction targets:

<div class="math-box">

- Predict $x_0$ · "x0-prediction" — the clean signal directly
- Predict $\mu_\theta(x_t, t)$ · the mean of the reverse distribution
- Predict $\epsilon$ · the noise that was added

</div>

Ho et al. 2020 showed **$\epsilon$-prediction gives the best sample quality**. Intuition · noise is unit-variance and dimension-independent; the network doesn't need to learn the scale of the signal.

Modern models (SDXL, Imagen) often use a fourth target, "v-prediction" — a fixed mix of $\epsilon$ and $x_0$ that's more numerically stable across timesteps.

---

# Training · the noise-guessing game

<div class="keypoint">

How do we teach the network to "un-blur"?

Take a clean image. Add a **known amount of random noise** $\epsilon$. Show the noisy result to the network. Ask it: *"What noise did I just add?"*

</div>

The better it gets at guessing the noise, the better it is at **denoising** · because subtracting the predicted noise gets us back to a cleaner image.

That's the entire training objective · MSE between predicted noise and the true noise added.

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
            x = mean + (1 - alpha_t).sqrt() * torch.randn_like(x)   # sigma_t = sqrt(beta_t)
        else:
            x = mean
    return x
```

The network architecture is a **U-Net** (L9) with time-step conditioning injected into each block.

---

# Reverse step · denoise then re-noise a little

To go from $x_t$ to $x_{t-1}$:
1. **Denoise.** Predict $\epsilon$, subtract a scaled version from $x_t$ → estimate of clean signal.
2. **Re-noise a little.** Add a small fresh noise so the chain stays stochastic.

$$x_{t-1} = \underbrace{\frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right)}_{\text{denoised mean}} + \underbrace{\sigma_t\,z}_{\text{small fresh noise}}$$

Why re-noise at all? Without it every chain from the same $x_T$ collapses to one deterministic path — the noise keeps sampling *stochastic*. Let's run one step by hand.

---

# Reverse step · worked numeric

**1D example.** $x_{100} = 1.5$. Schedule · $\alpha_{100} = 0.99$, $\bar\alpha_{100} = 0.8$. Network predicts $\epsilon_\theta = 0.6$.

- $1/\sqrt{0.99} \approx 1.005$
- $(1 - 0.99)/\sqrt{1 - 0.8} = 0.01 / \sqrt{0.2} \approx 0.0224$
- Mean · $1.005 \cdot (1.5 - 0.0224 \cdot 0.6) = 1.005 \cdot 1.4866 \approx \mathbf{1.494}$

Add noise · $\sigma_{100} = 0.1$, $z = -0.3$ → noise term = $-0.03$.
$x_{99} = 1.494 - 0.03 = \mathbf{1.464}$.

We took one small step from noisier (1.5) to slightly cleaner (1.464). At $t = 1$, drop the noise term — final step is deterministic.

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

*Optional depth (⭐⭐⭐) — examinable only for the curious.*

---

# The score field in one picture

![w:920px](figures/lec21/svg/score_field.svg)

---

# Score · the "uphill arrows" view

<div class="keypoint">

Imagine the probability density as a landscape. Real data sits on the **high peaks**; far from the data, the landscape is a **flat lowland**.

The **score** is an arrow at every point pointing **uphill** — toward higher density, i.e. toward real data.

</div>

If we learn this field of uphill arrows, we can generate · start anywhere in the lowlands (pure noise) and **follow the arrows uphill**, with a little random jitter, until we reach a peak (real data).

That's score-based generation. Mathematically equivalent to DDPM · just a different lens — gradients pointing toward data.

---

# The score function · math

Define $s(x) = \nabla_x \log p(x)$ — gradient of log density.
- $p(x)$ · density (high for real data, low elsewhere).
- $\log$ · just makes math nice (peaks of $p$ = peaks of $\log p$).
- $\nabla_x$ · vector pointing in the direction of steepest ascent.

If we have $s$, we can sample with **Langevin dynamics**:
$$x_{k+1} = x_k + \eta\,s(x_k) + \sqrt{2\eta}\,\xi,\quad \xi \sim \mathcal{N}(0, I)$$

A small step *toward* high-density regions, plus a bit of noise to keep exploring. Looks just like the reverse diffusion step · *follow a learned signal + add a little noise*.

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

# Diffusion ≈ score matching · derivation

The noisy distribution is Gaussian:
$x_t \sim \mathcal{N}(\mu = \sqrt{\bar\alpha_t}\,x_0,\ \sigma^2 = 1 - \bar\alpha_t)$

Log density (up to const):
$\log q(x_t \mid x_0) = -\dfrac{1}{2\sigma^2}(x_t - \mu)^2 + C$

Differentiate w.r.t. $x_t$:
$\nabla_{x_t}\log q = -\dfrac{1}{1 - \bar\alpha_t}(x_t - \sqrt{\bar\alpha_t}\,x_0)$

But the forward equation rearranges to $\epsilon = (x_t - \sqrt{\bar\alpha_t}\,x_0)/\sqrt{1 - \bar\alpha_t}$. Substitute:
$$\nabla_{x_t}\log q(x_t \mid x_0) = -\frac{\epsilon}{\sqrt{1 - \bar\alpha_t}}$$

The true score is just… the noise, negated and rescaled.

---

# The punchline · noise prediction *is* score estimation

<div class="insight">

The true score is just (negative, scaled) noise. So a network trained to predict $\epsilon$ with MSE is *already* a score model:

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar\alpha_t}}$$

</div>

DDPM (Ho 2020) and score-SDE (Song 2020) are two lenses on the **same** model — one trains "guess the noise", the other "learn the uphill arrows", and the equations coincide.

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

# Applications · the landscape today

- **Text-to-image** · Stable Diffusion, Midjourney, DALL-E 3, Imagen.
- **Video** · Sora, Runway Gen-3, VEO.
- **Audio** · AudioGen, Riffusion.
- **Molecule design** · RFdiffusion for proteins.
- **Robotics policies** · diffusion policy (Chi et al. 2023).

As of this writing, diffusion is the default generative model across modalities.

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

Consistency models and flow matching are closing the "slow sampling" gap — as of this writing, few-step samplers are already approaching GAN speed on many tasks.

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

# Putting it all together · the L21 master sentence

<div class="math-box">

**Diffusion = noise the data over $T$ steps · learn to denoise one step at a time.** The forward process $q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1}, \beta_t I)$ has a closed-form jump $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I)$ — a direct consequence of L00's "Gaussians stay Gaussian under linear ops". The training loss reduces to **predicting the noise** · $\mathcal{L} = \mathbb{E}_{x_0, t, \epsilon} \| \epsilon - \hat\epsilon_\theta(x_t, t) \|^2$. That's it · diffusion is **MSE on noise.**

</div>

| Object | Closed form | Where it came from |
|:-:|:-:|:-:|
| $q(x_t \mid x_0)$ | $\mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$ | sum of indep. Gaussians (L00) |
| Loss | $\| \epsilon - \hat\epsilon_\theta \|^2$ | NLL of Gaussian (L00) |
| Score $\nabla \log p_t(x)$ | $\propto \hat\epsilon$ | score matching (L21) |
| Sampling | reverse SDE / DDIM | next lecture (L22) |

**Pure L00 + an iterative MSE.** That's the whole theory.

---

# Practice problems

<div class="math-box">

**P1.** Verify the closed-form jump · show $q(x_t \mid x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t) I)$ where $\bar\alpha_t = \prod_{s \le t}(1 - \beta_s)$.

**P2.** Show that the simplified DDPM loss (predict $\epsilon$) is equivalent to a **weighted ELBO**. Why does training with the simplified objective work better than the full ELBO?

**P3.** Score $s_\theta(x, t) = \nabla_x \log p_t(x)$ vs noise $\hat\epsilon_\theta(x, t)$. Show $s_\theta(x_t, t) = -\hat\epsilon_\theta(x_t, t) / \sqrt{1 - \bar\alpha_t}$.

**P4.** A linear $\beta$-schedule from $10^{-4}$ to $0.02$ over $T = 1000$ steps. Compute $\bar\alpha_T$. What does this say about how thoroughly $x_T$ has been noised?

**P5.** **Why a U-Net for $\hat\epsilon_\theta$?** Connect to L09's segmentation argument · same image-in-image-out structure with skip connections preserves spatial fidelity.

**P6.** Compare diffusion vs GAN on **mode coverage**. Why does diffusion's training objective avoid GAN's mode collapse?

</div>

---

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

---

# The one-sentence takeaway

<div class="insight">

**Learn to denoise; generate by iterating.**

*Next lecture: the engineering that turns this recipe into Stable Diffusion.*

</div>
