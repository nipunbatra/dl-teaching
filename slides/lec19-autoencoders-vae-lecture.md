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

# Deriving the ELBO · one line at a time

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

<!-- _class: section-divider -->

### PART 4

# Reparameterization in one picture

![w:920px](figures/lec19/svg/reparam_trick.svg)

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

<!-- _class: section-divider -->

### PART 5

# Generating with a VAE

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
