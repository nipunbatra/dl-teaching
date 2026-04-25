---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# Autoencoders &amp; VAEs

## Lecture 19 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Learning outcomes

By the end of this lecture you will be able to:

1. State what a **plain autoencoder** is and why it isn't generative.
2. Motivate the need for a **prior distribution** on latent space.
3. Write the **ELBO** from scratch using Jensen's inequality.
4. Derive the **KL term** for Gaussian posterior vs standard normal.
5. Explain and implement the **reparameterization trick**.
6. Train a **β-VAE** and discuss disentanglement.
7. Place VAEs in context · pre-compressor for Stable Diffusion in 2026.

---

# Where we are

Module 9 opens · **generative models**. Until now every model *classified* or *predicted* — labels, tokens, pixels-given-labels. Today we switch to: **given a dataset, can I sample new examples that look like it?**

<div class="paper">

Today maps to **Prince Ch 17** (Variational Autoencoders) + Kingma &amp; Welling 2013.

</div>

Four questions:

1. What's a plain **autoencoder** and why isn't it generative?
2. What does **VAE** add, and why does it work?
3. What is the **reparameterization trick**?
4. Where do VAEs fit in the 2026 generative landscape?

---

# Generative modeling · the task

<div class="keypoint">

Given $N$ i.i.d. samples $\{x_1, \ldots, x_N\}$ from an unknown distribution $p_\text{data}$, learn a model from which we can **sample** new $x \sim p_\text{model}$ such that $p_\text{model} \approx p_\text{data}$.

</div>

Two sub-tasks, often pursued together:

1. **Density estimation** · assign a probability $p_\text{model}(x)$ to any candidate sample.
2. **Generation** · draw novel samples from $p_\text{model}$.

Images live in $\mathbb{R}^{100{,}000+}$ on a low-dimensional manifold. Writing down $p_\text{data}$ analytically is hopeless; learning it from samples is the whole game.

---

# Three generative strategies

<div class="columns">
<div>

### Density-based

Write $p_\theta(x)$ explicitly; maximize $\sum_i \log p_\theta(x_i)$.

- Pixel-RNN, PixelCNN, Normalizing flows.
- Clean likelihoods.
- Autoregressive or invertible only.

</div>
<div>

### Latent-based

Hidden variable $z$ generates $x$: $x \sim p(x|z), z \sim p(z)$.

- **VAE**, GAN (implicit).
- Compact latent, rich samples.
- Likelihood tractable only via ELBO.

</div>
</div>

Diffusion (L21) is a *layered* latent model with $T$ levels. The recipe of "add structure in latent space" starts here with the VAE.

---

<!-- _class: section-divider -->

### PART 1

# The generative model family tree

A brief taxonomy

---

# Four families of generative models

| Family | How it samples | Training |
|--------|----------------|----------|
| **VAE** (L19) | sample z ~ p(z), decode | ELBO |
| **GAN** (L20) | sample z ~ p(z), generator | minimax |
| **Normalizing flows** | invertible transforms of p(z) | exact likelihood |
| **Diffusion** (L21-22) | iterative denoising from noise | score matching / denoising |

Today is VAE. Each family has tradeoffs between sample quality, training stability, and tractability.

---

<!-- _class: section-divider -->

### PART 2

# Autoencoder first

The building block

---

# The plain autoencoder

<div class="math-box">

**Encoder** $f: \mathbb{R}^n \to \mathbb{R}^d$ · compresses input to a small latent code $z$.
**Decoder** $g: \mathbb{R}^d \to \mathbb{R}^n$ · reconstructs from $z$.

**Loss** · $\mathcal{L} = \|x - g(f(x))\|^2$

</div>

Train end-to-end. The bottleneck $d < n$ forces the network to learn a useful compression.

**Uses:**
- Denoising
- Dimensionality reduction (beats PCA for non-linear structure)
- Pretraining / feature learning

---

# Autoencoder vs PCA · what's added

PCA is **the** linear autoencoder with orthogonal weights. What does nonlinearity buy you?

<div class="math-box">

- PCA forces the latent space to be *linear subspace*. Fine for Gaussian-like data; poor for curved manifolds.
- An autoencoder (MLP or CNN) can fold arbitrary manifolds — digits on a swiss-roll latent, faces on a curved surface, etc.

</div>

Concretely · PCA on MNIST reaches ~85% explained variance with 32 dims; a deep AE matches the full-data reconstruction at ~16 dims. Curved manifold vs linear subspace · nonlinearity buys 2× compression.

---

# Bottleneck intuition · why it's crucial

If the latent $d \ge n$, the network can just copy · $z = x$, $g(z) = z$. Loss is zero but nothing learned.

<div class="keypoint">

The bottleneck $d \ll n$ **forces compression** · the network must keep only the most informative features. Anything redundant gets dropped. This is why autoencoders produce useful representations even without labels.

</div>

Modern variants add noise (denoising AE) or masking (MAE, L17) *instead of* a small bottleneck — same idea, different forcing.

---

# A concrete AE · MNIST dimensionality

<div class="math-box">

Input · `28 × 28 = 784` pixels. Encode to latent `z` of size 16. Decode back to 784.

| Layer | Shape | Params |
|:-:|:-:|:-:|
| Input | 784 | — |
| Linear → ReLU | 256 | 200,960 |
| Linear → ReLU | 64 | 16,448 |
| Linear (μ only) | 16 | **bottleneck** · 1,040 |
| Linear → ReLU | 64 | 1,088 |
| Linear → ReLU | 256 | 16,640 |
| Linear → sigmoid | 784 | 201,488 |

</div>

Total · ~440k params. Reconstruction MSE on MNIST test · ~0.003 after 10 epochs. Compare PCA with 16 components · ~0.015. **5× better with nonlinearities.**

---

# But autoencoders aren't generative

Suppose you train an AE on MNIST. To *generate* a new digit, you'd:

1. Pick a random z in latent space.
2. Decode it.

**What happens?** Usually garbage. Why?

<div class="warning">

The latent space is **irregular**. The encoder only learned to map *actual training images* to latent points. Random z values likely fall into "nothing-mapped-here" regions where the decoder is undefined.

</div>

You'd need the latent space to be **dense** and **structured** — that's what VAE adds.

---

<!-- _class: section-divider -->

### PART 3

# The VAE fix

A prior and a KL penalty

---

# AE vs VAE

![w:920px](figures/lec19/svg/ae_vs_vae.svg)

<div class="realworld">

▶ Interactive: slide the KL weight β, watch the latent space go from clumpy to Gaussian — [vae-latent-explorer](https://nipunbatra.github.io/interactive-articles/vae-latent-explorer/).

</div>

---

# Why a prior? · two jobs it does

The prior $p(z) = \mathcal{N}(0, I)$ does two things for us:

<div class="columns">
<div>

### 1. Defines the sampling distribution

At generation time we draw $z \sim p(z)$ and decode. The prior is the *rule book* for producing valid z's.

Without a prior, you wouldn't know how to initialize z for generation.

</div>
<div>

### 2. Regularizes the posterior

The KL term pulls $q(z|x)$ toward $p(z)$ for every training example. Every encoded posterior overlaps in the same region → smooth latent.

Without this, training points occupy disjoint clusters.

</div>
</div>

<div class="keypoint">

A VAE is a plain AE **with a regularizer that makes the latent space match a known distribution**. Everything else follows from making that regularizer principled (the ELBO).

</div>

---

# VAE · the encoder outputs a distribution

The encoder no longer outputs a point $z$. It outputs **parameters of a Gaussian**:

$$q_\phi(z | x) = \mathcal{N}(z; \mu_\phi(x), \sigma_\phi^2(x))$$

Both $\mu$ and $\sigma$ are network outputs. During training:

1. Given $x$, get $\mu(x), \sigma(x)$.
2. **Sample** $z \sim q_\phi(z|x)$.
3. Decode · $\hat{x} = g_\theta(z)$.
4. Loss · reconstruction **+** KL divergence to prior.

---

# ELBO geometry

![w:920px](figures/lec19/svg/elbo_geometry.svg)

---

# The punchline · you can skip the derivation

<div class="keypoint">

The VAE loss is just **reconstruction + KL-to-prior**. Train to minimize it. That's all you need to use a VAE.

$$\mathcal{L} = \underbrace{\|x - \text{decode}(z)\|^2}_\text{reconstruction} + \underbrace{\text{KL}(q(z|x)\,\|\,\mathcal{N}(0,I))}_\text{regularizer}$$

</div>

The next two slides *derive* this from first principles (Jensen's inequality). **If you trust me, you can skip them** · come back to the math later.

---

# ⚠️ optional · Deriving the ELBO · one line at a time

<div class="math-box">

Start from the log marginal:

$$\log p(x) = \log \int p(x, z)\, dz$$

Multiply and divide by $q(z|x)$ inside the integral:

$$= \log \int q(z|x) \cdot \frac{p(x, z)}{q(z|x)}\, dz = \log\, \mathbb{E}_{q}\!\left[\frac{p(x, z)}{q(z|x)}\right]$$

By Jensen's inequality (log is concave):

$$\geq \mathbb{E}_{q}\!\left[\log \frac{p(x, z)}{q(z|x)}\right] = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z|x)\,\|\,p(z))$$

That's the ELBO — a rigorous lower bound, tight when $q \to p(z|x)$.

</div>

---

# The ELBO · reconstruction + KL

<div class="math-box">

**Evidence Lower Bound** — a tractable lower bound on $\log p(x)$:

$$\log p(x) \geq \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{reconstruction}} \;-\; \underbrace{D_\text{KL}(q_\phi(z|x) \| p(z))}_{\text{regularization}}$$

- **First term** · data likelihood under the decoder, averaged over encoder samples.
- **Second term** · how far the posterior $q$ is from the prior $p(z) = \mathcal{N}(0, I)$.

</div>

Maximize the ELBO = minimize the negative. This is the VAE loss. Every term is tractable and backpropagatable.

---

# The KL term · in one line

For Gaussian $q$ and standard-normal prior:

<div class="math-box">

$$D_\text{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0, 1)) = \frac{1}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2 \right)$$

</div>

One expression, closed form, no sampling needed. Just plug in the encoder's $\mu$ and $\log \sigma^2$ outputs.

The KL is what **structures** the latent space · it pulls every encoded distribution toward the same standard normal, so random samples from N(0, I) land somewhere the decoder has seen.

---

# KL · worked numeric example

Suppose $q(z|x) = \mathcal{N}(\mu = 1.2, \sigma^2 = 0.25)$ for a particular image $x$. Standard normal prior $p(z) = \mathcal{N}(0, 1)$.

<div class="math-box">

$$\text{KL} = \frac{1}{2}(\sigma^2 + \mu^2 - 1 - \log \sigma^2)$$
$$= \frac{1}{2}(0.25 + 1.44 - 1 - \log 0.25) = \frac{1}{2}(0.69 + 1.386) = 1.038$$

</div>

Plugging in: $\sigma^2 = 0.25$ means **narrow** posterior (confident encoding); $\mu = 1.2$ means **off-centre** from the prior. The KL penalty of ~1.0 will pull $\mu$ back toward 0 during training, unless the reconstruction term needs a wide-apart $\mu$ to distinguish this image from others.

This is the trade-off the VAE balances at every sample.

---

# Posterior collapse · the picture

![w:920px](figures/lec19/svg/posterior_collapse.svg)

---

# Posterior collapse · what to watch for

If the decoder is too powerful, the KL term will drive $q(z|x) \to p(z) = \mathcal{N}(0, I)$ · every image encodes to the same latent, $z$ carries no information about $x$.

<div class="warning">

**Posterior collapse** · the VAE becomes an autoencoder where z is just noise. Reconstructions are fine (the decoder ignores z), but samples are junk (there's no latent structure to exploit).

</div>

**Fixes**:
- Reduce decoder capacity (smaller FFN).
- Use β-VAE with $\beta < 1$ (less KL weight).
- KL annealing · start with $\beta = 0$, ramp up over training.
- Free bits · allow some KL "for free" before penalizing.

---

<!-- _class: section-divider -->

### PART 4

# Reparameterization in one picture

![w:920px](figures/lec19/svg/reparam_trick.svg)

---

# Reparameterization · gradient flow

![w:920px](figures/lec19/svg/reparam_gradient_flow.svg)

---

# The reparameterization trick

How to backprop through a sample

---

# The problem

We need $z \sim \mathcal{N}(\mu, \sigma^2)$ for the decoder. But **sampling is not differentiable** — you can't backprop through a random draw.

<div class="keypoint">

**Kingma &amp; Welling 2013 trick** — move the randomness out of the path we want gradients through.

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Now $z$ is a *deterministic* function of $\mu$ and $\sigma$ (and the noise $\epsilon$, which has no parameters). Gradients flow through $\mu$ and $\sigma$ fine.

</div>

---

<!-- _class: code-heavy -->

# VAE in PyTorch · the whole thing

```python
class VAE(nn.Module):
    def __init__(self, d_in, d_z):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                 nn.Linear(256, 2 * d_z))
        self.dec = nn.Sequential(nn.Linear(d_z, 256), nn.ReLU(),
                                 nn.Linear(256, d_in))

    def forward(self, x):
        h = self.enc(x)
        mu, log_var = h.chunk(2, dim=-1)

        # Reparam: z = mu + sigma · eps
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps

        recon = self.dec(z)

        # KL against N(0, I)
        kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(-1)

        return recon, kl
```

Entire VAE in 15 lines. The trick is making sure the KL goes in the loss alongside reconstruction.

---

# Training loop

```python
for x in loader:
    recon, kl = model(x)
    recon_loss = F.mse_loss(recon, x, reduction='none').sum(-1)
    loss = (recon_loss + BETA * kl).mean()
    opt.zero_grad(); loss.backward(); opt.step()
```

Tuning `BETA`:
- `BETA = 1` · standard VAE (follows the ELBO derivation).
- `BETA > 1` · β-VAE (Higgins 2017). Stronger regularization → more disentangled, often blurrier.
- `BETA < 1` · more weight to reconstruction, less to latent structure.

---

# β · the seesaw between recon and KL

![w:920px](figures/lec19/svg/vae_loss_balance.svg)

---

# Disentanglement · what β-VAE buys you

With $\beta = 4$ on a faces dataset (Higgins 2017), each latent dimension starts to control ONE semantic factor:

<div class="math-box">

- $z_1$ · azimuth (face angle)
- $z_2$ · lighting direction
- $z_3$ · smile / frown
- $z_4$ · hairstyle length
- ...

</div>

<div class="insight">

No supervision — the structure emerges from the KL regularization plus the reconstruction pressure. Disentanglement lets you do **editable generation** · "same face with different smile" by perturbing one z coordinate.

</div>

The trade-off · stronger KL forces shared structure, but loses reconstruction detail. β = 1 is the theoretical sweet spot; higher β sacrifices quality for interpretability.

---

# Conditional VAE · putting labels into the game

If you have class labels $y$, a **Conditional VAE** extends the game:

<div class="math-box">

- Encoder: $q_\phi(z \mid x, y)$
- Decoder: $p_\theta(x \mid z, y)$
- Prior: $p(z) = \mathcal{N}(0, I)$ unchanged.

</div>

At inference · sample $z \sim \mathcal{N}(0, I)$, fix $y$ to the desired class, decode. Generate class-specific samples without retraining.

<div class="realworld">

CVAE was used for controllable generation before diffusion + CFG took over. Still shipped in some specialized systems (molecule generation, time-series imputation).

</div>

---

<!-- _class: section-divider -->

### PART 5

# Generating with a VAE

---

# Worked example · one VAE forward pass

Suppose · $d_x = 4$ (4-pixel input), $d_z = 2$ (2D latent). Input · $x = [1.0, 0.5, 0.2, 0.8]$.

<div class="math-box">

**Step 1 · encoder.** Suppose it outputs $\mu = [0.4, -0.3]$, $\log \sigma^2 = [-1.4, -2.0]$ → $\sigma = [0.5, 0.37]$.

**Step 2 · sample $\epsilon = [0.6, -0.2]$** (one draw from $\mathcal{N}(0, I)$).
$z = \mu + \sigma \odot \epsilon = [0.4 + 0.5 \cdot 0.6, -0.3 + 0.37 \cdot (-0.2)] = [0.70, -0.37]$

**Step 3 · decode.** Suppose decoder outputs $\hat x = [0.92, 0.45, 0.25, 0.81]$.

**Step 4 · loss.**
- Reconstruction MSE · $\|\hat x - x\|^2 = 0.008^2 + 0.05^2 + 0.05^2 + 0.01^2 \approx 0.005$
- KL · $\frac{1}{2} \sum (\sigma^2 + \mu^2 - 1 - \log \sigma^2)$
  - $z_1$: $0.25 + 0.16 - 1 + 1.4 = 0.81$
  - $z_2$: $0.137 + 0.09 - 1 + 2.0 = 1.23$
  - sum / 2 = **1.02**
- Total loss = 0.005 + 1.02 = **1.025**

</div>

Backprop through this. Update params. Repeat.

---

# Latent-space interpolation · in pictures

![w:920px](figures/lec19/svg/vae_interpolation.svg)

---

# Sampling + interpolation

```python
# Generate new examples
with torch.no_grad():
    z = torch.randn(16, d_z)              # sample from N(0, I)
    samples = model.dec(z)                # decode into image space

# Interpolate between two inputs in latent space
z_a = encoder(x_a)[0]  # mu for image A
z_b = encoder(x_b)[0]  # mu for image B
for alpha in torch.linspace(0, 1, 10):
    z = (1 - alpha) * z_a + alpha * z_b
    morph = model.dec(z)                  # smooth transition
```

The interpolation is the magic · it produces *valid* intermediate images because the latent space is smooth.

---

# Sampling gotchas

<div class="warning">

**Truncated sampling.** Sampling $z \sim \mathcal{N}(0, I)$ occasionally gives large-$\|z\|$ points where training coverage was sparse. Truncate · sample $z$ from $\mathcal{N}(0, I)$ and **reject** if $\|z\| > \tau$ (e.g., $\tau = 2.5$). Samples look cleaner.

**Decoder stochasticity.** If decoder outputs a Gaussian $p(x|z) = \mathcal{N}(g(z), \sigma_x^2 I)$, add $\sigma_x \cdot \epsilon_x$ to the mean for a single sample. If you only decode means you get the "mode"; adding variance makes samples diverse.

**VAE blur.** The KL pulls posteriors toward a simple prior · posteriors overlap significantly. The decoder averages over possible $z$ given $x$ → samples are blurry means. This is the fundamental VAE limitation diffusion (L21) fixes.

</div>

---

# VAE vs GAN vs Diffusion · quality ranking

| | Sample quality | Training stability | Likelihood | Sampling speed |
|--|---------------|---------------------|-----------|----------------|
| **VAE** | ✗ often blurry | ✓ stable | ✓ ELBO | ✓✓ one pass |
| **GAN** | ✓✓ sharp | ✗ brittle | ✗ no | ✓✓ one pass |
| **Diffusion** | ✓✓✓ SOTA | ✓ stable | ≈ | ✗ many passes |

<div class="insight">

VAEs remain useful for **latent-space exploration** and **pre-compression** — Stable Diffusion uses a VAE to compress images into a 4× smaller latent space *before* running diffusion there.

</div>

---

# Common questions · FAQ

**Q. Why is VAE blurrier than GAN?**
A. VAE's loss is MSE on pixels, which is the mean of possible reconstructions. When multiple outputs are possible (e.g., any detailed face), the mean is a smoothed average of those — blurry. GANs don't average; they commit.

**Q. Can I use a perceptual loss (feature-space MSE) instead of pixel MSE?**
A. Yes — produces sharper reconstructions. VQ-VAE (Van den Oord 2017) combines VAE-like structure with discrete latents and perceptual losses. Stable Diffusion's VAE uses this trick.

**Q. Is the posterior truly Gaussian?**
A. No — the *true* posterior is arbitrary. The Gaussian parameterization is an **approximation** (the "amortized variational" part). Normalizing flow encoders and hierarchical VAEs address this; vanilla VAE trades approximation quality for simplicity.

---

<!-- _class: summary-slide -->

# Lecture 19 — summary

- **Autoencoder** · encode → bottleneck → decode. Great for compression; *not* generative.
- **VAE** · encoder outputs a *distribution* (μ, σ); sample; decode; ELBO loss.
- **ELBO** · reconstruction term + KL divergence to prior.
- **KL term** · closed form for Gaussian; pulls q(z|x) toward N(0, I).
- **Reparameterization trick** · z = μ + σ·ε; differentiable.
- **β-VAE** · tune the KL weight for disentanglement.
- **2026 role** · pre-compressor in latent diffusion models (next week).

### Read before Lecture 20

Prince Ch 15 · GANs.

### Next lecture

**GANs** — minimax training, DCGAN, mode collapse, non-saturating loss.

<div class="notebook">

**Notebook 19** · `19-vae-mnist.ipynb` — build and train a VAE on MNIST; visualize 2D latent; interpolate digits; sample from N(0, I).

</div>
