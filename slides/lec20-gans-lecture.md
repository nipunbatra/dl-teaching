---
marp: true
theme: anthropic
paginate: true
math: mathjax
---

<!-- _class: title-slide -->

# GANs

## Lecture 20 · ES 667: Deep Learning

**Prof. Nipun Batra**
*IIT Gandhinagar · Aug 2026*

---

# Where we are

- **VAE** (L19) · probabilistic encoder, latent-space structure, blurry samples.

Today: **GANs**. Completely different philosophy — no likelihood, no prior, two networks duking it out.

<div class="paper">

Today maps to **Prince Ch 15** (GANs) + Goodfellow 2014 (original GAN) + Radford 2015 (DCGAN) + Arjovsky 2017 (WGAN).

</div>

Four questions:

1. What is the **minimax game**?
2. Why is GAN training so **unstable**?
3. What is **mode collapse** and how do we fight it?
4. What is **WGAN** and what did it fix?

---

<!-- _class: section-divider -->

### PART 1

# The minimax game

Two networks · adversarial training

---

# The GAN pipeline

![w:920px](figures/lec20/svg/gan_loop.svg)

---

# The minimax objective

<div class="math-box">

$$\min_G \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] \,+\, \mathbb{E}_{z \sim p(z)}[\log(1 - D(G(z)))]$$

**D** maximizes · get close to 1 on real data, close to 0 on generated data.
**G** minimizes · the second term — push $D(G(z))$ toward 1.

</div>

In theory, if both networks converge to Nash equilibrium, $G$ produces the exact data distribution and $D$ gives 0.5 on everything.

In practice, they never cleanly converge.

---

# Alternating updates · the training loop

```python
for batch in loader:
    # 1. Update discriminator
    opt_D.zero_grad()
    real = batch
    fake = G(torch.randn(BATCH, NOISE_DIM)).detach()   # stop grad through G here
    loss_D = -(D(real).log() + (1 - D(fake)).log()).mean()
    loss_D.backward(); opt_D.step()

    # 2. Update generator
    opt_G.zero_grad()
    fake = G(torch.randn(BATCH, NOISE_DIM))
    loss_G = -(D(fake).log()).mean()     # non-saturating (see next slide)
    loss_G.backward(); opt_G.step()
```

**Key pattern** — alternate between updating D and G. Balance is critical: if D gets too strong, G can't learn.

---

# The non-saturating trick

The original G objective — minimize $\log(1 - D(G(z)))$ — **saturates** when $D$ is confident about fake samples. Gradient goes to zero.

<div class="math-box">

**Non-saturating** G objective (Goodfellow 2014 footnote, became the standard):

$$\max_G\; \mathbb{E}_z[\log D(G(z))]$$

Maximize $\log D(G(z))$ instead of minimizing $\log(1 - D(G(z)))$. Same optimum, much better gradients early in training.

</div>

Everyone uses the non-saturating version.

---

<!-- _class: section-divider -->

### PART 2

# DCGAN · the architecture that worked

Radford et al. 2015

---

# DCGAN · five architectural guidelines

<div class="paper">

Radford, Metz, Chintala 2015 · "Unsupervised Representation Learning with Deep Convolutional GANs"

</div>

1. Replace pooling with **strided convolutions** (both D and G).
2. Use **batch normalization** in both.
3. Remove fully-connected hidden layers.
4. **ReLU** in G (except output, which uses Tanh).
5. **LeakyReLU** in D.

These aren't deep insights — they are a cookbook that made GANs actually train.

---

<!-- _class: code-heavy -->

# Generator in PyTorch · DCGAN

```python
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, 3, 4, 2, 1, bias=False),
            nn.Tanh()                                    # output in [-1, 1]
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))
```

Noise shape: `[batch, 100]` → reshaped to `[batch, 100, 1, 1]` → upsampled to `[batch, 3, 32, 32]`.

---

<!-- _class: section-divider -->

### PART 3

# Training instability &amp; mode collapse

The pathologies

---

# Why GANs are hard to train

GAN training is a **non-cooperative game**. Three common failure modes:

1. **D too strong** — confidently rejects all fakes early. G gets zero gradient, never improves.
2. **G too strong** — fools D completely. D loses discriminative power, G has no signal.
3. **Mode collapse** — G finds a few outputs that consistently fool D. Stops exploring.

<div class="warning">

Most of 2015–2019 GAN research is fighting these failure modes.

</div>

---

# Mode collapse visually

![w:920px](figures/lec20/svg/mode_collapse.svg)

---

# Diagnosing GAN health

Standard diagnostics:

- **D loss** · should hover ~0.5 (balanced). If D loss → 0, D is winning too much.
- **G loss** · should plateau, not grow.
- **Samples** · visually check for diversity. Do you see the same face repeatedly? Mode collapse.
- **Inception Score (IS)** · diversity + quality metric for image GANs.
- **FID (Fréchet Inception Distance)** · distance between fake and real feature distributions. Lower is better.

FID is the main quantitative metric for image generation in 2026.

---

<!-- _class: section-divider -->

### PART 4

# WGAN · Wasserstein distance

Arjovsky et al. 2017

---

# The WGAN insight

Original GAN minimizes **Jensen-Shannon divergence** — but JS saturates when distributions don't overlap. At the start of training, fake and real are very different → no gradient.

**WGAN replaces JS with the Wasserstein distance (Earth-Mover's Distance):**

<div class="math-box">

$$W(p_\text{real}, p_\text{fake}) = \inf_\gamma \mathbb{E}_{(x, y) \sim \gamma}[\|x - y\|]$$

Intuition: minimum "work" to transform one distribution into the other. Smooth even when distributions don't overlap.

</div>

---

# WGAN-GP · the practical version

Gulrajani et al. 2017 · replaces weight clipping (unstable) with a **gradient penalty**:

<div class="math-box">

$$\mathcal{L}_\text{WGAN-GP} = \mathbb{E}[D(G(z)) - D(x)] + \lambda\, \mathbb{E}_{\hat{x}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]$$

Enforces D to be **1-Lipschitz** — stabilizes training and eliminates mode collapse in practice.

</div>

---

<!-- _class: section-divider -->

### PART 5

# GANs in 2026

Still alive, but niche

---

# The GAN era · 2014-2020

| Year | Model | What it did |
|------|-------|-------------|
| 2014 | GAN | original paper — 28×28 MNIST |
| 2015 | DCGAN | convolutional, stable training |
| 2017 | WGAN-GP | Wasserstein + gradient penalty |
| 2017 | ProGAN | progressive growing → 1024×1024 faces |
| 2019 | StyleGAN | disentangled latent, hyper-realistic faces |
| 2021 | StyleGAN3 | temporal consistency, aliasing fixes |

---

# After GANs · diffusion took over

| | GANs | Diffusion |
|---|------|-----------|
| Training stability | ✗ brittle | ✓ stable |
| Sample quality | ✓ (SOTA 2018-2020) | ✓✓ (SOTA 2021+) |
| Likelihood | ✗ | ≈ |
| Inference speed | ✓✓ one pass | ✗ many passes |
| Diversity | mode collapse risk | ✓ natural |
| Latent space | rich, explorable | uniform-ish |

<div class="realworld">

In 2026 · **diffusion dominates** text-to-image / video. GANs survive where inference speed matters (real-time face generation, StyleGAN-based editing).

</div>

---

<!-- _class: summary-slide -->

# Lecture 20 — summary

- **GAN** · two networks, minimax objective · G generates, D classifies real vs fake.
- **Non-saturating G loss** · maximise $\log D(G(z))$ — better gradients early.
- **DCGAN** · architectural cookbook (stride convs, BN, ReLU/LeakyReLU, Tanh output).
- **Mode collapse** · G produces few distinct outputs; fight with diversity regularizers or WGAN.
- **WGAN-GP** · Wasserstein distance + gradient penalty · stable training.
- **2026** · diffusion has largely replaced GANs; StyleGAN still used where real-time generation matters.

### Read before Lecture 21

Prince Ch 18 · Diffusion models (early sections).

### Next lecture

**Diffusion Models — Theory** — forward noising, DDPM training, score matching.

<div class="notebook">

**Notebook 20** · `20-dcgan-mnist.ipynb` — DCGAN on MNIST or CelebA subset; monitor D/G loss balance; generate face grid.

</div>
